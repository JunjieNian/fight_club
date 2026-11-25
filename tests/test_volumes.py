"""Compare the GPU volume integrator with analytical homogeneous media."""

from pathlib import Path
import unittest

import numpy as np
import slangpy as spy

from renderer import gpu
from renderer.render import bind


class VolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.device = spy.create_device(include_paths=[str(root / "shaders")])
        cls.shader = gpu.load_compute_shader(cls.device, str(root / "tests" / "volume_checks.slang"))
        cls.sampler = cls.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear, mag_filter=spy.TextureFilteringMode.linear)
        cls.output = cls.device.create_texture(
            width=1, height=1, format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source)

    def integrate(self, densities, colors=None, limit=20, step=.3, origin=(0, 0, 3)):
        resolution = 4
        capacity = max(1, len(densities))
        colors = colors or [[0, 0, 0] for _ in densities]
        fields = np.zeros((resolution * capacity, resolution, resolution, 2), dtype=np.float32)
        lighting = np.zeros((*fields.shape[:3], 4), dtype=np.float32)
        volumes = []
        for index, (components, color) in enumerate(zip(densities, colors)):
            region = slice(index * resolution, (index + 1) * resolution)
            fields[region] = [components, 0] if np.isscalar(components) else components
            density = np.sum(components)
            lighting[region, ..., :3] = np.asarray(color) * density
            lighting[region, ..., 3] = density
            volumes.append({"center": [0, 0, 0], "half_extent": [1, 1, 1], "slot": index})
        density_texture = self.device.create_texture(
            type=spy.TextureType.texture_3d, width=resolution, height=resolution,
            depth=resolution * capacity, format=spy.Format.rg32_float,
            usage=spy.TextureUsage.shader_resource, data=fields)
        light_texture = self.device.create_texture(
            type=spy.TextureType.texture_3d, width=resolution, height=resolution,
            depth=resolution * capacity, format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource, data=lighting)
        values = {"num_explosions": len(volumes), "smoke_capacity": capacity,
                  "smoke_density_resolution": resolution, "smoke_light_resolution": resolution,
                  "smoke_step_size": step, "smoke_albedo": .75,
                  "explosions": gpu.create_structured_buffer(self.device, self.shader.module.Explosion, volumes),
                  "smoke_sampler": self.sampler, "smoke_density_grid": density_texture,
                  "smoke_lighting_grid": light_texture, "test_limit": limit,
                  "test_ray": {"origin": list(origin), "direction": [0, 0, -1]},
                  "test_background": [.2, .4, .6], "test_output": self.output}
        encoder = self.device.create_command_encoder()
        with encoder.begin_compute_pass() as compute_pass:
            root = compute_pass.bind_pipeline(self.shader.kernel.pipeline)
            bind(spy.ShaderCursor(root), values)
            compute_pass.dispatch([1, 1, 1])
        submission = self.device.submit_command_buffer(encoder.finish())
        self.device.wait_for_submit(submission)
        return self.output.to_numpy()[0, 0]

    def test_absorption_matches_beer_lambert_at_different_step_sizes(self):
        transmission = np.exp(-.4 * 2)
        expected = [*(np.asarray([.2, .4, .6]) * transmission), transmission]
        for step in (.8, .3, .05):
            with self.subTest(step=step):
                np.testing.assert_allclose(self.integrate([.4], step=step), expected, atol=2e-6)

    def test_constant_source_matches_the_closed_form_solution(self):
        transmission = np.exp(-.8)
        source = np.asarray([.1, .2, .3])
        expected = source * (1 - transmission) + np.asarray([.2, .4, .6]) * transmission
        np.testing.assert_allclose(self.integrate([.4], [source.tolist()])[:3], expected, atol=2e-6)

    def test_soot_and_dust_in_one_cell_add_their_extinction(self):
        np.testing.assert_allclose(self.integrate([[.13, .27]]), self.integrate([.4]), atol=2e-6)

    def test_opaque_surface_clips_the_integrated_path(self):
        result = self.integrate([.4], limit=2.5)
        transmission = np.exp(-.4 * .5)
        np.testing.assert_allclose(result[:3], np.asarray([.2, .4, .6]) * transmission, atol=2e-6)
        self.assertAlmostEqual(result[3], transmission, places=6)

    def test_foreground_surface_keeps_background_unchanged(self):
        np.testing.assert_allclose(self.integrate([.4], limit=1), [.2, .4, .6, 1], atol=1e-7)

    def test_empty_and_zero_density_media_preserve_background(self):
        for densities in ([], [0.0]):
            np.testing.assert_allclose(self.integrate(densities), [.2, .4, .6, 1], atol=1e-7)

    def test_overlapping_volumes_add_optical_depth_independent_of_order(self):
        colors = [[.15, 0, 0], [0, 0, .3]]
        first = self.integrate([.3, .6], colors)
        second = self.integrate([.6, .3], colors[::-1])
        transmission = np.exp(-1.8)
        source = (np.asarray(colors[0]) * .3 + np.asarray(colors[1]) * .6) / .9
        expected = source * (1 - transmission) + np.asarray([.2, .4, .6]) * transmission
        np.testing.assert_allclose(first[:3], expected, atol=2e-6)
        np.testing.assert_allclose(first, second, atol=2e-6)

    def test_parallel_ray_outside_the_volume_is_unaffected(self):
        np.testing.assert_allclose(self.integrate([.4], origin=(2, 0, 3)), [.2, .4, .6, 1], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
