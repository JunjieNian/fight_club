"""Build the self-contained environment scene for the final window shot."""

from pathlib import Path
import json
import math
import tempfile

import numpy as np
import trimesh


PROJECT_DIR = Path(__file__).resolve().parent


def write_json(path, payload, indent=None):
    contents = json.dumps(payload, indent=indent, separators=None if indent else (",", ":"))
    if path.exists() and path.read_text(encoding="utf-8") == contents:
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(contents)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_scene():
    scene_dir = PROJECT_DIR / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    geometry = {}
    materials = {}
    motions = {}
    buildings = {}
    rng = np.random.default_rng(167)

    def add(name, mesh, color, kind=0):
        geometry.setdefault(name, []).append(mesh)
        materials.setdefault(name, {"color": color, "kind": kind})

    def box(name, center, size, color, kind=0, rotation=0):
        mesh = trimesh.creation.box(extents=size)
        mesh.unmerge_vertices()
        mesh.vertex_normals = np.repeat(mesh.face_normals, 3, axis=0)
        if rotation:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(rotation), [0, 1, 0]))
        mesh.apply_translation(center)
        add(name, mesh, color, kind)

    def ellipsoid(name, center, radii, color, kind=0):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale(radii)
        mesh.apply_translation(center)
        add(name, mesh, color, kind)

    def rod(name, start, end, radius_start, radius_end, color):
        delta = np.asarray(end) - start
        length = float(np.linalg.norm(delta))
        rings = [(0, radius_start * .85), (length * .12, radius_start),
                 (length * .85, radius_end), (length, radius_end * .85)]
        segments = 12
        vertices = []
        for y, radius in rings:
            for j in range(segments):
                angle = j * 2 * math.pi / segments
                vertices.append([radius * math.cos(angle), y, radius * math.sin(angle)])
        faces = []
        for i in range(len(rings) - 1):
            for j in range(segments):
                a = i * segments + j
                b = (i + 1) * segments + j
                c = (i + 1) * segments + (j + 1) % segments
                d = i * segments + (j + 1) % segments
                faces.extend([[a, b, c], [a, c, d]])
        vertices.extend([[0, 0, 0], [0, length, 0]])
        for j in range(segments):
            faces.append([len(vertices) - 2, j, (j + 1) % segments])
            a = (len(rings) - 1) * segments + j
            b = (len(rings) - 1) * segments + (j + 1) % segments
            faces.append([len(vertices) - 1, b, a])
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        transform = trimesh.geometry.align_vectors([0, 1, 0], delta / length)
        transform[:3, 3] = start
        mesh.apply_transform(transform)
        add(name, mesh, color)

    # Interior dimensions are in metres. The city lies on the -Z side of the window.
    floor = [.045, .066, .073]
    frame = [.024, .033, .036]
    black = [.012, .017, .019]
    box("floor", (0, -.075, 4.1), (9, .15, 8.4), floor, kind=2)
    box("ceiling", (0, 3.72, 4.1), (9, .16, 8.4), [.19, .21, .21])
    box("back_wall", (0, 1.8, 8.38), (9, 3.6, .16), [.12, .15, .16])
    for x in (-2.70, 2.70):
        box("interior", (x, 1.8, .02), (.32, 3.6, .56), [.037, .061, .068])
    for x in (-3.65, -1.25, 0, 1.25, 3.65):
        box("window_frame", (x, 1.85, -.19), (.042, 3.7, .10), frame)
    for y in (.10, 2.15, 3.55):
        box("window_frame", (0, y, -.19), (8.2, .044, .14), frame)
    box("window_sill", (0, .025, -.10), (8.2, .07, .28), [.042, .060, .063])

    # Sparse office furniture and debris visible around the lower frame.
    chair_x, chair_z = -1.93, 1.85
    ellipsoid("chair", (chair_x, .40, chair_z), (.30, .055, .27), black)
    ellipsoid("chair", (chair_x, .64, chair_z + .16), (.29, .24, .065), black)
    rod("chair_metal", (chair_x, .12, chair_z),
         (chair_x, .41, chair_z), .027, .026, [.055, .068, .073])
    for j in range(5):
        angle = j * 2 * math.pi / 5
        end = (chair_x + .32 * math.cos(angle), .07,
               chair_z + .32 * math.sin(angle))
        rod("chair_metal", (chair_x, .13, chair_z), end, .018, .015, frame)
        ellipsoid("chair", (end[0], .045, end[2]), (.045, .037, .04), black)
    for side in (-1, 1):
        rod("chair_metal", (chair_x + side * .27, .40, chair_z + .08),
             (chair_x + side * .30, .57, chair_z + .04), .018, .018, frame)
        box("chair", (chair_x + side * .30, .57, chair_z - .02),
            (.07, .035, .26), black)
    box("equipment", (2.12, .32, 2.30), (1.80, .64, 1.0), [.020, .027, .030])
    box("equipment", (2.12, .67, 2.30), (1.87, .06, 1.06), [.030, .039, .043])
    box("equipment", (2.45, .86, 1.93), (.57, .30, .15), [.015, .020, .023], rotation=-9)
    box("equipment", (-2.95, .20, .47), (.60, .40, .72), [.038, .047, .048])
    rod("worklight", (-3.15, .03, .65), (-3.15, .90, .65),
        .014, .014, [.11, .085, .028])
    box("worklight", (-3.15, .91, .65), (.20, .10, .12), [.11, .085, .028])
    box("worklight_glow", (-3.15, .91, .72), (.15, .065, .012), [1.6, 1.2, .65], kind=3)
    for i in range(17):
        x = rng.uniform(-2.9, -1.12)
        z = rng.uniform(.25, 3.0)
        box("papers", (x, .008 + i * .0003, z),
            (rng.uniform(.12, .25), .002, rng.uniform(.10, .20)),
            [.21, .24, .23], rotation=float(rng.uniform(-75, 75)))
    for cable_index in range(3):
        points = []
        for i in range(30):
            t = i / 29
            points.append((-3.1 + 2.7 * t, .015,
                           .50 + cable_index * .14 + .22 * math.sin(t * 9)))
        for start, end in zip(points, points[1:]):
            rod("cables", start, end, .009, .009, black)

    ladder_start = np.array([2.92, .008, .75])
    ladder_end = np.array([2.30, 3.25, .34])
    for offset in (0, .40):
        rod("ladder", ladder_start + [offset, 0, 0], ladder_end + [offset, 0, 0],
            .028, .028, [.046, .051, .050])
    for rung in range(10):
        point = ladder_start + (ladder_end - ladder_start) * (.06 + rung * .095)
        rod("ladder", point, point + [.40, 0, 0], .019, .019, [.046, .051, .050])

    # Deeper towers extend farther below the window opening.
    towers = [(-23, -98, 14.5, 13, -24.5, 17), (-4, -108, 17, 16, -28, 20),
              (62, -168, 20, 18, -38.5, 14), (-65, -153, 18, 20, -35, 4),
              (95, -220, 23, 22, -49, 18)]
    for i, (x, z, w, d, base, roof) in enumerate(towers):
        h = roof - base
        color = [.026 + i * .002, .045, .055]
        name = f"tower_{i + 1}"
        section_count = 8 if i < 3 else 1
        buildings[name] = {"foundation": [x, base, z], "height": h,
                           "width": w, "depth": d,
                           "material_seed": [12, 14, 15, 16, 17][i]}
        boundaries = base + np.rint(np.linspace(0, h / 1.75, section_count + 1)) * 1.75
        boundaries[-1] = base + h
        for level in range(section_count):
            section_name = f"{name}_section_{level:02d}" if section_count > 1 else name
            bottom, top = boundaries[level:level + 2]
            section_height = top - bottom
            pivot = [x, (bottom + top) / 2, z]
            box(section_name, pivot, (w, section_height, d), color, kind=1)
            materials[section_name]["seed"] = [12, 14, 15, 16, 17][i]
            if section_count > 1:
                motions[section_name] = {"building": name, "level": level,
                                        "pivot": pivot, "height": section_height,
                                        "ground": base, "building_height": h,
                                        "seed": 167 + i * 100 + level}
        box(section_name, (x, base + h - .6, z), (w + .55, 1.2, d + .55), color, kind=1)
        box(section_name, (x + w * .19, base + h + 1.1, z - d * .1),
            (w * .28, 2.2, d * .32), color, kind=1)
        beacon_name = f"{name}_beacon" if section_count > 1 else "roof_lights"
        box(beacon_name, (x + w * .19, base + h + 2.28, z - d * .1),
            (.20, .18, .20), [.65, .025, .012], kind=3)
        if section_count > 1:
            motions[beacon_name] = motions[section_name]

    for i in range(52):
        z = -float(rng.uniform(145, 420))
        x = float(rng.uniform(-185, 185))
        w = float(rng.uniform(7, 19))
        d = float(rng.uniform(9, 22))
        h = float(rng.uniform(31, 93))
        box("distant_city", (x, -94 + h / 2, z), (w, h, d),
            [.025, .042, .052], kind=1)
    materials["distant_city"]["seed"] = 18
    box("city_ground", (0, -94.5, -235), (700, 1, 550), [.013, .025, .031])

    objects = []
    meshes = []
    triangle_count = 0
    for name, parts in geometry.items():
        mesh = trimesh.util.concatenate(parts)
        filename = f"{name}.json"
        payload = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32).round(6).tolist(),
            "faces": np.asarray(mesh.faces, dtype=np.int32).tolist(),
            "vertex_normals": np.asarray(mesh.vertex_normals, dtype=np.float32).round(6).tolist(),
            "vertex_colors": np.ones((len(mesh.vertices), 3), dtype=np.float32).tolist(),
        }
        write_json(scene_dir / filename, payload)
        color = materials[name]["color"]
        objects.append({"name": name, "type": "mesh", "index": len(meshes),
                        "filename": filename, "Kd": color, "Kr": [0, 0, 0],
                        "kind": materials[name]["kind"],
                        "material_seed": materials[name].get("seed", len(objects)),
                        "material": {"Kd": color, "Ks": [.05, .05, .05], "exponent": 32}})
        if name in motions:
            objects[-1]["animation"] = motions[name]
        meshes.append({"filename": filename})
        triangle_count += len(mesh.faces)

    fov_x = 63
    width, height = 1920, 800
    aspect = width / height
    camera = {"resolution": [width, height], "fov_x": fov_x,
              "s": math.tan(math.radians(fov_x / 2)) / aspect,
              "z_near": .03,
              "transform": [{"lookat": {"position": [0, 1.10, 6.2],
                                         "target": [0, 1.18, -35], "up": [0, 1, 0]}}]}
    scene = {"camera": camera, "background": [.010, .027, .040],
             "buildings": buildings,
             "duration": 15, "max_depth": 2, "meshes": meshes, "objects": objects,
             "lights": [{"position": [-2.6, 4, -3], "intensity": [5, 12, 16]},
                        {"position": [2.3, 4.2, -4], "intensity": [5, 12, 16]},
                        {"position": [2.5, 3, 3], "intensity": [.3, .6, .8]},
                        {"position": [-3.05, .85, .78], "intensity": [1.4, .95, .40]}]}
    scene_path = scene_dir / "scene.json"
    write_json(scene_path, scene, indent=2)
    return scene_path, {"objects": len(objects), "triangles": triangle_count}
