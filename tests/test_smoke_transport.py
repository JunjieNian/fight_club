"""Check when, where, and how the rendered smoke moves on the GPU."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import slangpy as spy

from fight_club_scene import build_scene
from renderer.animation import load_shot, window_row_y
from renderer import gpu
from renderer.render import SceneRenderer, bind


class SmokeTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shot = load_shot(Path(__file__).resolve().parents[1] / "shot.json")
        cls.scene, _ = build_scene()
        cls.renderer = SceneRenderer(cls.scene, 96, 40, shot)
        cls.res = cls.renderer.shared["smoke_density_resolution"]
        domain = cls.renderer.smoke_domains[0]
        cls.event = shot["explosions"][0]
        cls.start = cls.event["time"]
        unit = (np.arange(cls.res) + .5) * 2 / cls.res - 1
        x, cls.y, z = [center + unit * extent for center, extent in zip(domain["center"], domain["half_extent"])]
        x = np.maximum(0, abs(x - domain["foundation"][0]) - domain["half_size"][0])
        z = np.maximum(0, abs(z - domain["foundation"][2]) - domain["half_size"][1])
        cls.distance_from_wall = np.hypot(z[:, None], x)

    def optical_field(self, age):
        self.renderer.render_frame(self.start + age)
        return self.renderer.shared["smoke_density_grid"].to_numpy()[:self.res]

    def mean_height(self, density):
        return np.average(self.y, weights=density.sum(axis=(0, 2)))

    def mean_wall_distance(self, density):
        return np.average(self.distance_from_wall, weights=density.sum(axis=1))

    def test_first_row_emits_soot_before_the_collapse_and_other_towers_stay_clear(self):
        field = self.optical_field(.12)
        building = self.renderer.scene_data["buildings"][self.event["building"]]
        y = window_row_y(building, self.event["floors"][0])
        self.assertGreater(field[..., 0].max(), 0)
        self.assertLess(abs(self.mean_height(field[..., 0]) - y), 2)
        self.assertEqual(field[..., 1].max(), 0)
        self.assertEqual(self.optical_field(self.event["collapse_delay"])[..., 1].max(), 0)
        raw = self.renderer.shared["transported_smoke"].to_numpy()
        self.assertEqual(raw[self.res:].max(), 0)

    def test_soot_rises_and_the_later_dust_spreads_below_it(self):
        early = self.optical_field(.75)
        mid = self.optical_field(2.5)
        late = self.optical_field(3.67)
        self.assertTrue(np.isfinite(late).all())
        self.assertGreaterEqual(late.min(), 0)
        self.assertGreater(self.mean_height(mid[..., 0]), self.mean_height(early[..., 0]) + 2)
        self.assertGreater(self.mean_height(late[..., 0]), self.mean_height(late[..., 1]) + 3)
        self.assertGreater(self.mean_wall_distance(late[..., 1]), self.mean_wall_distance(mid[..., 1]) + 1)

    def test_fractional_seek_reproduces_the_density_and_pixels(self):
        t = self.start + 1.37
        first = self.renderer.render_frame(t)
        field = self.renderer.shared["smoke_density_grid"].to_numpy()[:self.res].copy()
        self.renderer.render_frame(t + 1.21)
        self.renderer.render_frame(t - .24)
        repeated = self.renderer.render_frame(t)
        np.testing.assert_array_equal(field, self.renderer.shared["smoke_density_grid"].to_numpy()[:self.res])
        np.testing.assert_array_equal(first, repeated)

    def test_oldest_plume_retains_dense_cores_and_clear_sightlines_at_the_end(self):
        renderer = self.renderer
        renderer.render_frame(renderer.shot["duration"] - 1 / 12)
        shader = gpu.load_compute_shader(renderer.device, str(Path(__file__).with_name("visible_smoke.slang")),
                                         "measure_main")
        output = renderer.device.create_texture(
            width=renderer.width, height=renderer.height, format=spy.Format.rg32_float,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source)
        encoder = renderer.device.create_command_encoder()
        with encoder.begin_compute_pass() as compute_pass:
            root = compute_pass.bind_pipeline(shader.kernel.pipeline)
            # Isolate the oldest plume while all opaque geometry stays at its final pose.
            bind(spy.ShaderCursor(root), {**renderer.shared, "num_explosions": 1, "measured": output})
            compute_pass.dispatch([renderer.width, renderer.height, 1])
        renderer.device.wait_for_submit(renderer.device.submit_command_buffer(encoder.finish()))
        result = output.to_numpy()
        transmission = result[..., 0][result[..., 1] > .5]
        self.assertGreater(np.mean(transmission < .25), .10)
        self.assertGreater(np.mean(transmission > .65), .15)

    def test_second_scatter_lights_smoke_without_changing_extinction(self):
        self.renderer.render_frame(self.start + 2.5)
        res = self.renderer.shared["smoke_light_resolution"]
        direct = self.renderer.shared["smoke_direct_grid"].to_numpy()[:res]
        total = self.renderer.shared["smoke_lighting_grid"].to_numpy()[:res]
        np.testing.assert_array_equal(direct[..., 3], total[..., 3])
        self.assertTrue(np.all(total[..., :3] >= direct[..., :3]))
        self.assertGreater((total[..., :3] - direct[..., :3]).max(), .0001)
        np.testing.assert_array_equal(total[direct[..., 3] == 0], 0)

    def test_windows_on_a_stationary_tower_produce_only_soot(self):
        shot = deepcopy(self.renderer.shot)
        shot["explosions"] = [dict(self.event, compaction_ratio=1, lean_degrees=[0, 0])]
        renderer = SceneRenderer(self.scene, 48, 20, shot)
        renderer.render_frame(self.start + 2.5)
        field = renderer.shared["smoke_density_grid"].to_numpy()
        self.assertGreater(field[..., 0].max(), 0)
        self.assertEqual(field[..., 1].max(), 0)
