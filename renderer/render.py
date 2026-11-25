"""Ray tracing and display conversion for the fight_club animation."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import slangpy as spy

from renderer import gpu, scene
from renderer.animation import DEFAULT_EFFECTS, evaluate_frame, smoke_layout


PROJECT_DIR = Path(__file__).resolve().parents[1]
SMOKE_HZ = 12


def bind(cursor, values):
    for name, value in values.items():
        field = cursor.find_field(name)
        if field.is_valid():
            field.write(value)


class SceneRenderer:
    """Keep geometry and object-space BVHs resident while evaluating shot frames."""

    def __init__(self, scene_path, width, height, shot):
        self.width, self.height, self.shot = width, height, shot
        self.device = spy.create_device(include_paths=[str(PROJECT_DIR / "shaders")])
        shader_path = str(PROJECT_DIR / "shaders" / "fight_club.slang")
        self.compute = gpu.load_compute_shader(self.device, shader_path)
        bindings, self.scene_data, _ = scene.upload_scene(str(scene_path), self.compute.module, self.device)
        self.camera = bindings["camera"]
        self.shared = {name: bindings[name] for name in (
            "shapes", "num_shapes", "spheres", "meshes", "vertices", "vertex_normals",
            "triangles", "bvh_nodes", "lights", "num_lights")}
        objects = self.scene_data["objects"]
        for name, values, dtype in (
            ("material_kinds", [obj["kind"] for obj in objects], "uint"),
            ("material_seeds", [obj.get("material_seed", i) for i, obj in enumerate(objects)], "uint"),
            ("window_intensities", np.ones(len(objects)), "float")):
            array_type = np.uint32 if dtype == "uint" else np.float32
            self.shared[name] = gpu.structured_typed_buffer_from_numpy(self.device, np.asarray(values, dtype=array_type), dtype)
        self.shared["resolution"] = spy.float2(width, height)
        self.shared["explosions"] = gpu.create_structured_buffer(self.device, self.compute.module.Explosion, [])
        self.shared["num_explosions"] = 0
        self.shared["floor_fires"] = gpu.create_structured_buffer(self.device, self.compute.module.FloorFire, [])
        self.shared["num_floor_fires"] = 0
        self.smoke_domains, rows = smoke_layout(self.scene_data, shot)
        self.shared["smoke_capacity"] = max(1, len(self.smoke_domains))
        self.shared["smoke_density_resolution"] = 112
        self.shared["smoke_light_resolution"] = 64
        self.shared["smoke_step_size"] = DEFAULT_EFFECTS["smoke_step_size"]
        self.shared["smoke_albedo"] = DEFAULT_EFFECTS["smoke_albedo"]
        self.shared["smoke_sampler"] = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear, mag_filter=spy.TextureFilteringMode.linear)
        def volume_texture(resolution, texture_format):
            return self.device.create_texture(
                type=spy.TextureType.texture_3d, width=resolution, height=resolution,
                depth=resolution * self.shared["smoke_capacity"], format=texture_format,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
                      | spy.TextureUsage.copy_source)
        resolution = self.shared["smoke_density_resolution"]
        self.shared["smoke_density_grid"] = volume_texture(resolution, spy.Format.rg32_float)
        for name in ("smoke_direct_grid", "smoke_lighting_grid"):
            self.shared[name] = volume_texture(self.shared["smoke_light_resolution"], spy.Format.rgba32_float)
        self.transport_grids = [volume_texture(resolution, spy.Format.rg32_float) for _ in range(2)]
        self.prediction_grid = volume_texture(resolution, spy.Format.rg32_float)
        self.transport = gpu.load_compute_shader(self.device, str(PROJECT_DIR / "shaders" / "smoke_transport.slang"),
                                                 "advect_main")
        self.transport_sources = {
            "explosions": gpu.create_structured_buffer(self.device, self.transport.module.Explosion, self.smoke_domains),
            "num_explosions": len(self.smoke_domains),
            "smoke_rows": gpu.create_structured_buffer(self.device, self.transport.module.SmokeRow, rows)}
        self.smoke_start = min((domain["start_time"] for domain in self.smoke_domains), default=0)
        self.smoke_tick = 0
        self.transport_settings = None
        smoke_path = str(PROJECT_DIR / "shaders" / "smoke_fields.slang")
        self.smoke_density = gpu.load_compute_shader(self.device, smoke_path, "density_main")
        self.smoke_lighting = gpu.load_compute_shader(self.device, smoke_path, "lighting_main")
        self.smoke_scatter = gpu.load_compute_shader(self.device, smoke_path, "scatter_main")
        self.color = self.device.create_texture(
            width=width, height=height, format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource
                  | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source)

    def _transport_smoke(self, encoder, time_seconds, effects):
        # Commit only full fixed steps. A fractional preview uses the spare grid.
        position = (time_seconds - self.smoke_start) * SMOKE_HZ
        if round(position) > 0 and abs(position - round(position)) < 1e-8:
            position = round(position)
        target = int(position)
        half_lives = (effects["soot_half_life"], effects["dust_half_life"])
        settings = (effects["smoke_density"], effects["smoke_lift"], tuple(effects["smoke_wind"]), half_lives)
        tick = self.smoke_tick if self.smoke_tick <= target and self.transport_settings == settings else 0
        self.smoke_tick = 0  # A failed command submission must replay from the empty state.
        self.transport_settings = settings
        resolution = self.shared["smoke_density_resolution"]
        params = {**self.transport_sources, "smoke_capacity": self.shared["smoke_capacity"],
                  "smoke_density_resolution": resolution, "smoke_sampler": self.shared["smoke_sampler"],
                  "smoke_strength": effects["smoke_density"], "smoke_lift": effects["smoke_lift"],
                  "smoke_wind": effects["smoke_wind"], "smoke_half_lives": list(half_lives)}
        result = self.transport_grids[tick % 2]
        while tick < position:
            end = min(tick + 1, position)
            previous = self.transport_grids[tick % 2]
            result = self.transport_grids[(tick + 1) % 2]
            step = {**params, "simulation_time": self.smoke_start + end / SMOKE_HZ,
                    "step_seconds": (end - tick) / SMOKE_HZ, "history_valid": int(tick > 0),
                    "previous_smoke": previous}
            for predict, field, output in ((1, previous, self.prediction_grid), (0, self.prediction_grid, result)):
                with encoder.begin_compute_pass() as compute_pass:
                    root = compute_pass.bind_pipeline(self.transport.kernel.pipeline)
                    bind(spy.ShaderCursor(root), {**step, "predict_only": predict,
                                                 "predicted_smoke": field, "next_smoke": output})
                    compute_pass.dispatch([resolution, resolution, resolution * len(self.smoke_domains)])
                encoder.global_barrier()
            if end > target:
                break
            tick += 1
        return result, target

    def render_frame(self, time_seconds):
        state = evaluate_frame(self.scene_data, self.shot, time_seconds, self.smoke_domains)
        time_seconds = float(np.clip(time_seconds, 0, self.shot["duration"]))
        self.camera = state["camera"]
        for name, data_type in (("shapes", self.compute.module.Shape),
                               ("lights", self.compute.module.PointLight),
                               ("explosions", self.compute.module.Explosion),
                               ("floor_fires", self.compute.module.FloorFire)):
            self.shared[name] = gpu.create_structured_buffer(self.device, data_type, state[name])
        self.shared["num_lights"] = len(state["lights"])
        self.shared["num_explosions"] = len(state["explosions"])
        self.shared["num_floor_fires"] = len(state["floor_fires"])
        effects = {**DEFAULT_EFFECTS, **self.shot.get("effects", {})}
        self.shared["smoke_albedo"] = effects["smoke_albedo"]
        self.shared["smoke_step_size"] = effects["smoke_step_size"]
        self.shared.update(simulation_time=time_seconds, smoke_wind=effects["smoke_wind"],
                           smoke_detail=effects["smoke_detail"], smoke_lift=effects["smoke_lift"])
        self.shared["window_intensities"] = gpu.structured_typed_buffer_from_numpy(
            self.device, state["window_intensities"], "float")
        camera_matrix = np.asarray(self.camera["cam_to_world"], dtype=np.float32)
        self.shared["camera"] = {"cam_to_world": spy.float4x4(camera_matrix), "fov_x": float(self.camera["fov_x"])}
        encoder = self.device.create_command_encoder()
        smoke_tick = self.smoke_tick
        if self.shared["num_explosions"]:
            self.shared["transported_smoke"], smoke_tick = self._transport_smoke(encoder, time_seconds, effects)
            for shader, resolution, output_name, texture_name in (
                (self.smoke_density, self.shared["smoke_density_resolution"], "density_output", "smoke_density_grid"),
                (self.smoke_lighting, self.shared["smoke_light_resolution"], "lighting_output", "smoke_direct_grid"),
                (self.smoke_scatter, self.shared["smoke_light_resolution"], "lighting_output", "smoke_lighting_grid")):
                with encoder.begin_compute_pass() as compute_pass:
                    root = compute_pass.bind_pipeline(shader.kernel.pipeline)
                    bind(spy.ShaderCursor(root), {**self.shared, output_name: self.shared[texture_name]})
                    compute_pass.dispatch([resolution, resolution, resolution * self.shared["num_explosions"]])
                encoder.global_barrier()
        with encoder.begin_compute_pass() as compute_pass:
            root = compute_pass.bind_pipeline(self.compute.kernel.pipeline)
            bind(spy.ShaderCursor(root), {**self.shared, "output_image": self.color})
            compute_pass.dispatch([self.width, self.height, 1])
        submission = self.device.submit_command_buffer(encoder.finish())
        self.device.wait_for_submit(submission)
        self.smoke_tick = smoke_tick
        result = self.color.to_numpy().copy()
        if not np.isfinite(result).all():
            raise RuntimeError("The rendered frame contains non-finite values.")
        return result


def frame_pixels(frame, width, height):
    # Integrate supersamples in linear light before display conversion and bloom.
    channels = [np.asarray(Image.fromarray(frame[:, :, channel]).resize(
        (width, height), Image.Resampling.LANCZOS)) for channel in range(3)]
    rgb = np.maximum(np.stack(channels, axis=2), 0)
    brightness = np.max(rgb, axis=2, keepdims=True)
    highlights = rgb * np.maximum(brightness - .22, 0) / np.maximum(brightness, .001)
    rgb += gaussian_filter(highlights, sigma=(2.6, 2.6, 0)) * .24
    rgb += gaussian_filter(highlights, sigma=(12, 12, 0)) * .10
    y, x = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    radius = ((x / rgb.shape[1] - .5) / .75) ** 2 + ((y / rgb.shape[0] - .5) / .8) ** 2
    rgb *= (1 - .18 * np.clip(radius, 0, 1))[:, :, None]
    rgb = rgb / (1 + rgb)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)], axis=2)
    display_rgb = np.clip(rgba[:, :, :3], 0, 1)
    display_rgb = np.where(display_rgb <= .0031308, 12.92 * display_rgb,
                           1.055 * np.power(display_rgb, 1 / 2.4) - .055)
    display_rgba = np.concatenate([display_rgb, rgba[:, :, 3:4]], axis=2)
    return np.round(np.clip(display_rgba, 0, 1) * 255).astype(np.uint8)
