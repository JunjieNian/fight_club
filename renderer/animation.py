"""Evaluate the camera, floor fires, and connected vertical collapse at any time."""

import json
from pathlib import Path

import numpy as np

from .scene import compose_transformations, get_transformation, SPHERE, TRIANGLE_MESH


DEFAULT_EFFECTS = {
    "floor_radiance": [130000.0, 48000.0, 9600.0],
    "window_emission": 90.0,
    "flash_decay": 1.65,
    "smoke_density": .55,
    "soot_half_life": 10.0,
    "dust_half_life": 6.8,
    "smoke_radius": 28.0,
    "smoke_lift": 2.2,
    "smoke_detail": .18,
    "smoke_albedo": .75,
    "smoke_wind": [.55, 0.0, .18],
    "smoke_step_size": .5,
}
DEFAULT_FLOORS = [{"height": .30, "delay": 0.0},
                  {"height": .40, "delay": .16},
                  {"height": .48, "delay": .32}]
FLOOR_PITCH = 1.75
WINDOW_HALF_HEIGHT = FLOOR_PITCH * .23


def load_shot(path):
    shot = json.loads(Path(path).read_text(encoding="utf-8"))
    if not np.isfinite(shot["duration"]) or shot["duration"] <= 0:
        raise ValueError("Shot duration must be finite and positive.")
    keys = shot["camera_keyframes"]
    times = np.array([key["time"] for key in keys], dtype=float)
    if len(keys) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Camera keyframes need strictly increasing, finite times.")
    for key in keys:
        vectors = np.asarray([key["position"], key["target"]], dtype=float)
        if vectors.shape != (2, 3) or not np.isfinite(vectors).all():
            raise ValueError("Camera positions and targets must be finite 3D vectors.")
        if np.linalg.norm(vectors[0] - vectors[1]) < 1e-5 or not 0 < key["fov_x"] < 180:
            raise ValueError("Camera direction and field of view must be valid.")
    names = [event["building"] for event in shot["explosions"]]
    if len(set(names)) != len(names):
        raise ValueError("Each building must have one explosion event.")
    for event in shot["explosions"]:
        values = [event["time"], event["collapse_duration"],
                  event.get("collapse_delay", .85), event.get("intensity_scale", 1)]
        if not np.isfinite(values).all() or min(values) < 0 or event["collapse_duration"] <= 0:
            raise ValueError("Explosion timing and intensity must be finite and valid.")
        ratio = event.get("compaction_ratio", .08)
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("Compaction ratio must be greater than zero and at most one.")
        lean = np.asarray(event.get("lean_degrees", [.3, -.6]))
        if lean.shape != (2,) or not np.isfinite(lean).all():
            raise ValueError("Building lean must contain two finite angles.")
        floors = event.get("floors", DEFAULT_FLOORS)
        if not floors:
            raise ValueError("Each explosion needs at least one burning floor.")
        heights = [floor["height"] for floor in floors]
        if len(set(heights)) != len(heights):
            raise ValueError("Burning floors must have distinct heights.")
        for floor in floors:
            values = [floor["height"], floor["delay"]]
            if not np.isfinite(values).all() or not 0 < values[0] < 1 or values[1] < 0:
                raise ValueError("Floor heights must lie inside the building and delays must be nonnegative.")
    effects = {**DEFAULT_EFFECTS, **shot.get("effects", {})}
    if np.asarray(effects["floor_radiance"]).shape != (3,):
        raise ValueError("Floor radiance must be an RGB vector.")
    values = [*effects["floor_radiance"], effects["window_emission"], effects["flash_decay"], effects["smoke_density"],
              effects["smoke_radius"], effects["smoke_lift"], effects["smoke_detail"],
              effects["smoke_albedo"], effects["smoke_step_size"]]
    if not np.isfinite(values).all() or min(values) < 0 or min(effects["flash_decay"], effects["smoke_radius"]) <= 0:
        raise ValueError("Effects must be finite and nonnegative, with positive decay and radius.")
    if effects["smoke_albedo"] > 1 or min(effects["smoke_step_size"], effects["smoke_detail"]) <= 0:
        raise ValueError("Smoke albedo must be at most one, with positive detail frequency and step size.")
    half_lives = [effects["soot_half_life"], effects["dust_half_life"]]
    if not np.isfinite(half_lives).all() or min(half_lives) <= 0:
        raise ValueError("Smoke half-lives must be finite, positive seconds.")
    wind = np.asarray(effects["smoke_wind"], dtype=float)
    if wind.shape != (3,) or not np.isfinite(wind).all():
        raise ValueError("Smoke wind must be a finite 3D velocity.")
    shot["effects"] = effects
    return shot


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def camera_at(shot, time_seconds):
    keys = shot["camera_keyframes"]
    t = np.clip(time_seconds, keys[0]["time"], keys[-1]["time"])
    right = min(int(np.searchsorted([key["time"] for key in keys], t, side="right")), len(keys) - 1)
    left = max(0, right - 1)
    first, second = keys[left], keys[right]
    weight = float(smoothstep((t - first["time"]) / (second["time"] - first["time"])))
    position = (1 - weight) * np.asarray(first["position"]) + weight * np.asarray(second["position"])
    target = (1 - weight) * np.asarray(first["target"]) + weight * np.asarray(second["target"])
    if np.linalg.norm(np.cross(target - position, [0, 1, 0])) < 1e-6:
        raise ValueError("The interpolated camera direction must have a valid horizontal component.")
    matrix = compose_transformations([{"lookat": {"position": position, "target": target, "up": [0, 1, 0]}}])
    return {"cam_to_world": matrix, "fov_x": (1 - weight) * first["fov_x"] + weight * second["fov_x"]}


def section_transform(motion, event, time_seconds):
    age = max(0.0, time_seconds - event["time"] - event.get("collapse_delay", .85))
    if age == 0:
        return np.eye(4, dtype=np.float32)
    progress = min(1.0, age / event["collapse_duration"])
    height = motion["building_height"]
    front = height * progress * progress
    ratio = event.get("compaction_ratio", .08)
    pivot = np.asarray(motion["pivot"], dtype=float)
    ground = motion["ground"]
    bottom = pivot[1] - motion["height"] / 2 - ground
    top = bottom + motion["height"]
    # One continuous height map keeps neighboring section boundaries connected.
    rest_bounds = np.asarray([bottom, top])
    fallen_bounds = ratio * np.minimum(rest_bounds, front) + np.maximum(rest_bounds - front, 0)
    scale = (fallen_bounds[1] - fallen_bounds[0]) / motion["height"]
    center = [pivot[0], ground + fallen_bounds.mean(), pivot[2]]
    foundation = np.asarray([pivot[0], ground, pivot[2]])
    lean = np.asarray(event.get("lean_degrees", [.3, -.6])) * smoothstep(progress)
    return compose_transformations([
        {"translate": -pivot}, {"scale": [1, scale, 1]}, {"translate": center},
        {"translate": -foundation}, {"rotate_x": lean[0]}, {"rotate_z": lean[1]},
        {"translate": foundation}])


def window_row_y(building, floor):
    height = building["foundation"][1] + building["height"] * floor["height"]
    return float((np.floor(height / FLOOR_PITCH) + .53) * FLOOR_PITCH)


def smoke_layout(scene_data, shot):
    """Keep transported smoke in world space as its window sources descend."""
    effects = {**DEFAULT_EFFECTS, **shot.get("effects", {})}
    radius = effects["smoke_radius"]
    domains, rows = [], []
    for slot, event in enumerate(shot["explosions"]):
        building = scene_data["buildings"][event["building"]]
        x, ground, z = building["foundation"]
        sections = [obj["animation"] for obj in scene_data["objects"]
                    if obj.get("animation", {}).get("building") == event["building"]]
        first = len(rows)
        for floor in sorted(event.get("floors", DEFAULT_FLOORS), key=lambda row: row["height"]):
            y = window_row_y(building, floor)
            bottom, top = ground, ground + building["height"]
            for section in sections:
                low = section["pivot"][1] - section["height"] / 2
                if low <= y <= low + section["height"]:
                    bottom, top = low, low + section["height"]
                    break
            rows.append({"height": y - ground, "bottom": bottom - ground,
                         "top": top - ground, "delay": floor["delay"]})
        half_width, half_depth = building["width"] / 2, building["depth"] / 2
        low = ground - 2
        high = max(ground + building["height"] + 6, ground + rows[-1]["height"] + 1.4 * radius)
        domains.append({"center": [x, (low + high) / 2, z],
                        "half_extent": [half_width + radius, (high - low) / 2, half_depth + radius],
                        "foundation": [x, ground, z], "half_size": [half_width, half_depth],
                        "building_height": building["height"], "slot": slot,
                        "first_row": first, "row_count": len(rows) - first,
                        "start_time": event["time"], "seed": float(event["seed"]),
                        "collapse_delay": event.get("collapse_delay", .85),
                        "collapse_duration": event["collapse_duration"],
                        "compaction": event.get("compaction_ratio", .08) if sections else 1.0,
                        "lean": np.deg2rad(event.get("lean_degrees", [.3, -.6])).tolist() if sections else [0, 0]})
    return domains, rows


def floor_sources(building, motions, event, effects, time_seconds):
    """Build four rectangular facade emitters for each active window row."""
    x, _, z = building["foundation"]
    width, depth = building["width"], building["depth"]
    sources = []
    for floor in event.get("floors", DEFAULT_FLOORS):
        rest_y = window_row_y(building, floor)
        age = time_seconds - event["time"] - floor["delay"]
        if age <= 0 or age >= 7:
            continue
        pulse = (1 - np.exp(-age * 30)) * (.88 * np.exp(-age * effects["flash_decay"])
                                             + .12 * np.exp(-age * .45))
        pulse *= 1 - smoothstep((age - 4) / 3)
        scale = event.get("intensity_scale", 1)
        radiance = np.asarray(effects["floor_radiance"]) * pulse * scale
        glow = float(effects["window_emission"] * pulse * scale)
        if np.max(radiance) < .01 and glow < .001:
            continue
        motion = next((motion for motion in motions
                       if abs(rest_y - motion["pivot"][1]) < motion["height"] / 2), None)
        matrix = section_transform(motion, event, time_seconds) if motion else np.eye(4)
        # Pane coverage is integrated as an open-area fraction during illumination.
        for normal, tangent, half_width, half_depth in (
            ([0, 0, 1], [1, 0, 0], width / 2, depth / 2),
            ([0, 0, -1], [1, 0, 0], width / 2, depth / 2),
            ([1, 0, 0], [0, 0, 1], depth / 2, width / 2),
            ([-1, 0, 0], [0, 0, 1], depth / 2, width / 2),
        ):
            normal = np.asarray(normal, dtype=float)
            rest_center = np.asarray([x, rest_y, z]) + normal * (half_depth + .012)
            world_normal = np.linalg.inv(matrix[:3, :3]).T @ normal
            world_normal /= np.linalg.norm(world_normal)
            sources.append({"center": (matrix @ [*rest_center, 1])[:3].tolist(),
                            "half_u": (matrix[:3, :3] @ (np.asarray(tangent) * half_width)).tolist(),
                            "half_v": (matrix[:3, :3] @ [0, WINDOW_HALF_HEIGHT, 0]).tolist(),
                            "normal": world_normal.tolist(), "rest_normal": normal.tolist(),
                            "radiance": radiance.tolist(),
                            "glow": glow,
                            "rest_y": float(rest_y), "material_seed": building["material_seed"]})
    return sources


def evaluate_frame(scene_data, shot, time_seconds, smoke_domains=None):
    if not np.isfinite(time_seconds):
        raise ValueError("Shot time must be finite.")
    t = float(np.clip(time_seconds, 0, shot["duration"]))
    events = {event["building"]: event for event in shot["explosions"]}
    shapes = []
    sphere_index = 0
    window_intensities = []
    building_motions = {}
    for obj in scene_data["objects"]:
        matrix = get_transformation(obj, 0)
        intensity = 1.0
        motion = obj.get("animation")
        if motion and motion["building"] in events:
            event = events[motion["building"]]
            matrix = section_transform(motion, event, t) @ matrix
            intensity = float(np.exp(-max(0, t - event["time"] - .25) * .75))
            building_motions.setdefault(motion["building"], []).append(motion)
        is_sphere = obj["type"] == "sphere"
        shapes.append({"type": SPHERE if is_sphere else TRIANGLE_MESH,
                       "index": sphere_index if is_sphere else obj["index"],
                       "Kd": obj["Kd"], "Kr": obj["Kr"],
                       "obj_to_world": matrix, "world_to_obj": np.linalg.inv(matrix)})
        sphere_index += int(is_sphere)
        window_intensities.append(intensity)

    if smoke_domains is None:
        smoke_domains, _ = smoke_layout(scene_data, shot)
    volumes = [domain for domain in smoke_domains if t > domain["start_time"]]
    fires = []
    effects = {**DEFAULT_EFFECTS, **shot.get("effects", {})}
    for event in shot["explosions"]:
        age = t - event["time"]
        if age < 0 or age >= 12:
            continue
        building = scene_data["buildings"][event["building"]]
        fires.extend(floor_sources(building, building_motions.get(event["building"], []),
                                   event, effects, t))
    return {"camera": camera_at(shot, t), "shapes": shapes, "lights": list(scene_data["lights"]),
            "floor_fires": fires, "explosions": volumes,
            "window_intensities": np.asarray(window_intensities, dtype=np.float32)}
