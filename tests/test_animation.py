"""Regression checks for burning floors and connected vertical collapse."""

import json
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import trimesh

from fight_club_scene import build_scene
from renderer.animation import WINDOW_HALF_HEIGHT, camera_at, evaluate_frame, load_shot, section_transform, smoke_layout
from renderer.scene import get_transformation


class AnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot = load_shot(Path(__file__).resolve().parents[1] / "shot.json")
        scene_path, cls.counts = build_scene()
        cls.scene = json.loads(scene_path.read_text(encoding="utf-8"))

    def test_initial_frame_matches_rest_transforms(self):
        state = evaluate_frame(self.scene, self.shot, 0)
        np.testing.assert_array_equal(state["camera"]["cam_to_world"], get_transformation(self.scene["camera"], 0))
        for obj, shape in zip(self.scene["objects"], state["shapes"]):
            np.testing.assert_array_equal(shape["obj_to_world"], get_transformation(obj, 0))
        self.assertEqual(state["explosions"], [])
        self.assertEqual(state["floor_fires"], [])
        np.testing.assert_array_equal(state["window_intensities"], 1)

    def test_explosions_trigger_in_order(self):
        for time_seconds, count in ((2.9, 0), (3.2, 1), (6.2, 2), (8.7, 3)):
            with self.subTest(time=time_seconds):
                state = evaluate_frame(self.scene, self.shot, time_seconds)
                self.assertEqual(len(state["explosions"]), count)
                self.assertEqual(state["lights"], self.scene["lights"])

    def test_sections_fall_after_the_shared_collapse_delay(self):
        index = next(i for i, obj in enumerate(self.scene["objects"]) if obj["name"] == "tower_1_section_07")
        obj = self.scene["objects"][index]
        early = evaluate_frame(self.scene, self.shot, 3.7)["shapes"][index]["obj_to_world"]
        np.testing.assert_array_equal(early, np.eye(4))
        later = evaluate_frame(self.scene, self.shot, 5)["shapes"][index]["obj_to_world"]
        pivot = np.array([*obj["animation"]["pivot"], 1])
        self.assertLess((later @ pivot)[1], pivot[1])

    def test_roof_beacon_follows_top_section(self):
        state = evaluate_frame(self.scene, self.shot, 7)
        values = {obj["name"]: shape["obj_to_world"] for obj, shape in zip(self.scene["objects"], state["shapes"])}
        np.testing.assert_array_equal(values["tower_1_beacon"], values["tower_1_section_07"])

    def test_evaluation_is_independent_of_frame_order(self):
        first = evaluate_frame(self.scene, self.shot, 4.2)
        evaluate_frame(self.scene, self.shot, 12)
        repeated = evaluate_frame(self.scene, self.shot, 4.2)
        for a, b in zip(first["shapes"], repeated["shapes"]):
            np.testing.assert_array_equal(a["obj_to_world"], b["obj_to_world"])
        np.testing.assert_array_equal(first["camera"]["cam_to_world"], repeated["camera"]["cam_to_world"])
        self.assertEqual(first["explosions"], repeated["explosions"])
        self.assertEqual(first["floor_fires"], repeated["floor_fires"])

    def test_camera_endpoints_and_shot_clamping(self):
        for t, key in ((0, self.shot["camera_keyframes"][0]), (15, self.shot["camera_keyframes"][-1])):
            np.testing.assert_allclose(camera_at(self.shot, t)["cam_to_world"][:3, 3], key["position"])
        final = evaluate_frame(self.scene, self.shot, 15)
        held = evaluate_frame(self.scene, self.shot, 100)
        for a, b in zip(final["shapes"], held["shapes"]):
            np.testing.assert_array_equal(a["obj_to_world"], b["obj_to_world"])

    def test_transforms_remain_finite_and_invertible(self):
        for t in (0, 3.2, 6.5, 9, 15):
            state = evaluate_frame(self.scene, self.shot, t)
            for shape in state["shapes"]:
                self.assertTrue(np.isfinite(shape["obj_to_world"]).all())
                np.testing.assert_allclose(shape["obj_to_world"] @ shape["world_to_obj"], np.eye(4), atol=3e-5)

    def test_three_floor_fires_stay_around_the_middle_and_lower_floors(self):
        for event in self.shot["explosions"]:
            building = self.scene["buildings"][event["building"]]
            state = evaluate_frame(self.scene, self.shot, event["time"] + .65)
            fires = [fire for fire in state["floor_fires"] if fire["material_seed"] == building["material_seed"]]
            self.assertEqual(len(fires), 12)
            self.assertEqual(len({fire["rest_y"] for fire in fires}), 3)
            for fire in fires:
                fraction = (fire["rest_y"] - building["foundation"][1]) / building["height"]
                self.assertGreater(fraction, .25)
                self.assertLess(fraction, .60)
                self.assertLess(fire["center"][1], 0)
                self.assertGreater(fire["radiance"][0], fire["radiance"][2])

    def test_floor_ignition_is_staggered_and_fades(self):
        seed = self.scene["buildings"]["tower_1"]["material_seed"]
        for age, count in ((.1, 4), (.25, 8), (.6, 12)):
            fires = evaluate_frame(self.scene, self.shot, 3 + age)["floor_fires"]
            self.assertEqual(len([fire for fire in fires if fire["material_seed"] == seed]), count)
        early = evaluate_frame(self.scene, self.shot, 3.2)["floor_fires"][0]
        later = evaluate_frame(self.scene, self.shot, 4.2)["floor_fires"][0]
        self.assertGreater(early["radiance"][0], later["radiance"][0])

    def test_two_floor_events_are_supported(self):
        shot = deepcopy(self.shot)
        shot["explosions"][0]["floors"].pop()
        fires = evaluate_frame(self.scene, shot, 3.65)["floor_fires"]
        self.assertEqual(len(fires), 8)
        self.assertEqual(len({fire["rest_y"] for fire in fires}), 2)

    def test_burning_floor_centers_are_visible_through_the_window(self):
        for event in self.shot["explosions"]:
            building = self.scene["buildings"][event["building"]]
            state = evaluate_frame(self.scene, self.shot, event["time"] + .65)
            camera = state["camera"]["cam_to_world"][:3, 3]
            for fire in state["floor_fires"]:
                if fire["material_seed"] != building["material_seed"] or fire["rest_normal"] != [0, 0, 1]:
                    continue
                direction = np.asarray(fire["center"]) - camera
                crossing = camera + direction * ((-.19 - camera[2]) / direction[2])
                self.assertGreater(crossing[1], .14)
                self.assertLess(crossing[1], 3.52)
                self.assertLess(abs(crossing[0]), 4.1)

    def test_city_ground_and_foundations_stay_below_the_window_sill(self):
        root = Path(__file__).resolve().parents[1]
        ground_objects = [obj for obj in self.scene["objects"]
                          if obj["name"].startswith("city_") and obj["kind"] == 0]
        for t in np.linspace(0, self.shot["duration"], 31):
            state = evaluate_frame(self.scene, self.shot, t)
            camera = state["camera"]["cam_to_world"][:3, 3]
            for obj in ground_objects:
                mesh = json.loads((root / "scene" / obj["filename"]).read_text(encoding="utf-8"))
                vertices = np.asarray(mesh["vertices"])
                farthest_z = vertices[:, 2].min()
                highest_y = vertices[:, 1].max()
                crossing_y = camera[1] + (highest_y - camera[1]) * ((-.19 - camera[2]) / (farthest_z - camera[2]))
                self.assertLess(crossing_y, .10)
            for building in self.scene["buildings"].values():
                center = np.asarray(building["foundation"])
                center[2] -= building["depth"] / 2
                crossing_y = camera[1] + (center[1] - camera[1]) * ((-.19 - camera[2]) / (center[2] - camera[2]))
                self.assertLess(crossing_y, .10)
            if t == self.shot["duration"]:
                for obj, shape in zip(self.scene["objects"], state["shapes"]):
                    if not obj.get("animation"):
                        continue
                    mesh = json.loads((root / "scene" / obj["filename"]).read_text(encoding="utf-8"))
                    vertices = np.asarray(mesh["vertices"])
                    points = np.column_stack([vertices, np.ones(len(vertices))]) @ shape["obj_to_world"].T
                    crossing_y = camera[1] + (points[:, 1] - camera[1]) * ((-.19 - camera[2]) / (points[:, 2] - camera[2]))
                    self.assertLess(crossing_y.max(), .10)

    def test_ladder_clears_room_geometry_and_rests_at_floor_height(self):
        root = Path(__file__).resolve().parents[1] / "scene"
        components = {}
        for obj in self.scene["objects"]:
            if obj["name"] not in ("ladder", "interior", "window_frame", "window_sill", "equipment", "ceiling"):
                continue
            mesh = json.loads((root / obj["filename"]).read_text(encoding="utf-8"))
            geometry = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=True)
            components[obj["name"]] = geometry.split(only_watertight=False)
        floor_clearance = min(part.bounds[0, 1] for part in components["ladder"])
        self.assertGreaterEqual(floor_clearance, 0)
        self.assertLess(floor_clearance, .01)
        for ladder in components.pop("ladder"):
            for name, obstacles in components.items():
                for obstacle in obstacles:
                    overlap = np.minimum(ladder.bounds[1], obstacle.bounds[1]) - np.maximum(ladder.bounds[0], obstacle.bounds[0])
                    self.assertTrue(np.any(overlap <= 0), f"Ladder overlaps {name}.")

    def test_emitters_have_finite_area_and_consistent_normals(self):
        for t in (3.5, 5, 8, 10, 13):
            for fire in evaluate_frame(self.scene, self.shot, t)["floor_fires"]:
                cross = np.cross(fire["half_u"], fire["half_v"])
                self.assertGreater(np.linalg.norm(cross), 0)
                self.assertAlmostEqual(np.linalg.norm(fire["normal"]), 1)
                self.assertLess(np.linalg.norm(np.cross(cross, fire["normal"])), 1e-5)

    def test_floor_emitters_follow_their_collapsing_sections(self):
        state = evaluate_frame(self.scene, self.shot, 5.2)
        for fire in state["floor_fires"]:
            building_name, building = next((name, value) for name, value in self.scene["buildings"].items()
                                           if value["material_seed"] == fire["material_seed"])
            obj, shape = next((obj, shape) for obj, shape in zip(self.scene["objects"], state["shapes"])
                              if obj.get("animation", {}).get("building") == building_name
                              and abs(fire["rest_y"] - obj["animation"]["pivot"][1]) < obj["animation"]["height"] / 2)
            normal = np.asarray(fire["rest_normal"])
            depth = building["width"] if normal[0] else building["depth"]
            x, _, z = building["foundation"]
            center = [x, fire["rest_y"], z] + normal * (depth / 2 + .012)
            np.testing.assert_allclose((shape["obj_to_world"] @ [*center, 1])[:3], fire["center"], atol=1e-5)
            np.testing.assert_allclose(shape["obj_to_world"][:3, :3] @ [0, WINDOW_HALF_HEIGHT, 0], fire["half_v"], atol=1e-6)
            bottom = obj["animation"]["pivot"][1] - obj["animation"]["height"] / 2
            self.assertGreater(fire["rest_y"] - WINDOW_HALF_HEIGHT, bottom)
            self.assertLess(fire["rest_y"] + WINDOW_HALF_HEIGHT, bottom + obj["animation"]["height"])

    def test_smoke_sources_match_the_animated_window_sections_and_ignition_times(self):
        domains, rows = smoke_layout(self.scene, self.shot)
        for event, domain in zip(self.shot["explosions"], domains):
            ground = domain["foundation"][1]
            state = evaluate_frame(self.scene, self.shot, event["time"] + .65)
            building = self.scene["buildings"][event["building"]]
            floors = sorted({fire["rest_y"] for fire in state["floor_fires"]
                             if fire["material_seed"] == building["material_seed"]})
            start = domain["first_row"]
            sources = rows[start:start + domain["row_count"]]
            np.testing.assert_allclose([row["height"] + ground for row in sources], floors)
            self.assertEqual([row["delay"] for row in sources], [floor["delay"] for floor in event["floors"]])
            for row in sources:
                section = next(obj["animation"] for obj in self.scene["objects"]
                               if obj.get("animation", {}).get("building") == event["building"]
                               and abs(obj["animation"]["pivot"][1] - ground - row["height"])
                               < obj["animation"]["height"] / 2)
                self.assertAlmostEqual(row["bottom"] + ground, section["pivot"][1] - section["height"] / 2)
                self.assertAlmostEqual(row["top"] - row["bottom"], section["height"])

    def test_per_event_intensity_scales_floor_radiance(self):
        modified = deepcopy(self.shot)
        modified["explosions"][0]["intensity_scale"] *= .5
        full = evaluate_frame(self.scene, self.shot, 3.2)
        half = evaluate_frame(self.scene, modified, 3.2)
        for first, second in zip(full["floor_fires"], half["floor_fires"]):
            np.testing.assert_allclose(np.asarray(first["radiance"]) * .5, second["radiance"])
            self.assertEqual(first["glow"] * .5, second["glow"])
            self.assertEqual(first["center"], second["center"])
        self.assertEqual(full["explosions"], half["explosions"])

    def test_legacy_shots_use_effect_defaults(self):
        legacy = deepcopy(self.shot)
        legacy.pop("effects")
        expected = evaluate_frame(self.scene, self.shot, 3.2)
        actual = evaluate_frame(self.scene, legacy, 3.2)
        self.assertEqual(expected["lights"], actual["lights"])
        self.assertEqual(expected["explosions"], actual["explosions"])
        self.assertEqual(expected["floor_fires"], actual["floor_fires"])

    def test_section_boundaries_remain_connected(self):
        for event in self.shot["explosions"]:
            sections = [obj["animation"] for obj in self.scene["objects"]
                        if obj["name"].startswith(event["building"] + "_section_")]
            for age in np.linspace(0, 7, 21):
                t = event["time"] + age
                for lower, upper in zip(sections, sections[1:]):
                    x, y, z = lower["pivot"]
                    for dx, dz in ((0, 0), (-2, 2), (2, -2)):
                        boundary = [x + dx, y + lower["height"] / 2, z + dz, 1]
                        np.testing.assert_allclose(section_transform(lower, event, t) @ boundary,
                                                   section_transform(upper, event, t) @ boundary, atol=1e-5)

    def test_collapse_stays_vertical_with_positive_compaction(self):
        for event in self.shot["explosions"]:
            building = self.scene["buildings"][event["building"]]
            for age in np.linspace(0, 7, 21):
                for obj in self.scene["objects"]:
                    motion = obj.get("animation")
                    if not motion or motion["building"] != event["building"]:
                        continue
                    matrix = section_transform(motion, event, event["time"] + age)
                    pivot = np.asarray([*motion["pivot"], 1])
                    center = matrix @ pivot
                    self.assertGreater(np.linalg.det(matrix[:3, :3]), .079)
                    self.assertLess(np.linalg.norm((center - pivot)[[0, 2]]), building["height"] * .02)
                    self.assertLessEqual(center[1], pivot[1] + .01)

    def test_completed_towers_compact_at_their_foundations(self):
        for event in self.shot["explosions"]:
            top = next(obj["animation"] for obj in self.scene["objects"]
                       if obj["name"] == event["building"] + "_section_07")
            matrix = section_transform(top, event, 30)
            roof = [top["pivot"][0], top["ground"] + top["building_height"], top["pivot"][2], 1]
            actual_height = (matrix @ roof)[1] - top["ground"]
            self.assertAlmostEqual(actual_height, top["building_height"] * event["compaction_ratio"], delta=.01)


if __name__ == "__main__":
    unittest.main()
