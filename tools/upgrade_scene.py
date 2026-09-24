"""Apply the visually audited v2 geometry to the reference scene map."""
import json
from pathlib import Path

path = Path("src/scene.json")
scene = json.loads(path.read_text())
scene["notes"] = ("Coordinates in C3905 1920x1080 reference. Crosswalks and curb/line "
                  "positions audited against all four registered backgrounds and 4K crops. "
                  "No lane arrows or turn-prohibition signs are legible; manoeuvre permissions remain unknown.")
scene["signals"] = {
    "veh_main": {"roi": [1124, 352, 1180, 426], "type": "vehicle_3lamp",
                 "controls": "near", "comment": "Median-side front-facing signal; red/amber/green for near approach."},
    "left_red_green": {"roi": [249, 502, 291, 552], "type": "red_green_2lamp",
                 "controls": "near (probable)",
                 "comment": "Left-edge head turns red during median-head amber; use median red for vehicle red-light labels."},
    "gantry_left_back": {"roi": [368, 280, 399, 330], "type": "signal_back",
                         "controls": None, "comment": "Back of overhead signal; phase not visible."},
    "gantry_right_back": {"roi": [716, 253, 750, 311], "type": "signal_back",
                          "controls": None, "comment": "Back of overhead signal; phase not visible."},
}
scene["stop_lines"]["near"]["line"] = [[289, 545], [934, 466]]
scene["stop_lines"]["near"]["signal"] = "veh_main"
scene["stop_lines"]["near"]["comment"] = "Across all near approach lanes; front bumper defines crossing."
scene["zones"]["junction"]["polygon"] = [
    [285, 545], [934, 466], [1188, 517], [1920, 440], [1920, 1080],
    [610, 1080], [180, 787], [335, 630]]
scene["zones"]["junction"]["comment"] = "Full visible conflict area from near stop line through side-road and crossing B/C."
scene["solid_lines"] = {
    "near_curb_lane_end": {"polyline": [[100, 292], [171, 355], [276, 440], [430, 506]],
                           "comment": "Near-side lane boundary becomes continuous on stop-line approach; partly occluded."},
    "near_median_edge": {"polyline": [[120, 149], [401, 248], [780, 392], [1178, 514]],
                         "comment": "Yellow edge line along raised median; no vehicle crossing."},
}
scene["dashed_lines"] = {
    "near_lane_1": {"polyline": [[120, 235], [260, 297], [420, 372], [650, 474]],
                    "comment": "Approximate centreline of dashed near-side lane marking; last metres occluded by vehicles."},
    "near_lane_2": {"polyline": [[160, 185], [335, 255], [580, 352], [834, 459]],
                    "comment": "Dashed boundary of median-side near lane."},
    "far_lane_1": {"polyline": [[120, 105], [380, 162], [770, 269], [1190, 374], [1600, 463]],
                   "comment": "Far-carriageway dashed lane divider, approximate through heavy occlusion."},
    "far_lane_2": {"polyline": [[105, 135], [380, 213], [710, 307], [1070, 400], [1430, 479]],
                   "comment": "Second far-carriageway dashed divider, approximate."},
}
scene["curbs"] = {
    "near_left": {"polyline": [[67, 310], [181, 409], [282, 545], [337, 740], [180, 787]],
                  "comment": "Near-side curb and sidewalk edge."},
    "near_median": {"polyline": [[120, 141], [330, 213], [697, 353], [1150, 510], [1193, 523]],
                    "comment": "Raised central median edge."},
    "far_outer": {"polyline": [[95, 83], [420, 118], [900, 212], [1370, 322], [1815, 412], [1920, 433]],
                  "comment": "Far-side sidewalk curb, partly hidden by trees."},
    "island_B": {"polyline": [[1187, 515], [1305, 544], [1315, 566], [1193, 556]],
                 "comment": "Keep-right island between crossings A and B."},
    "island_C_inner": {"polyline": [[495, 787], [621, 686], [761, 765], [505, 797]],
                       "comment": "First pedestrian refuge island by lower crossing."},
    "island_C_outer": {"polyline": [[679, 886], [746, 828], [900, 855], [915, 889], [701, 920]],
                       "comment": "Second pedestrian refuge island by lower crossing."},
}
scene["no_stopping"] = {
    "near_crossing_A": {"polygon": [[285, 545], [934, 466], [1175, 530], [335, 630]],
                        "basis": "stop line plus zebra crossing", "comment": "Only red-light stop before the line is permitted."},
    "branch_crossing_B": {"polygon": [[1195, 510], [1840, 460], [1853, 494], [1220, 547]],
                          "basis": "zebra crossing"},
}
scene["lanes"] = {
    "near_curb": {"polygon": [[70, 310], [140, 250], [420, 505], [289, 545]],
                  "direction": [0.88, 0.47], "allowed_manoeuvres": "unknown",
                  "comment": "No legible turn arrow or restriction sign in samples."},
    "near_middle": {"polygon": [[140, 250], [185, 195], [705, 475], [420, 505]],
                    "direction": [0.9, 0.43], "allowed_manoeuvres": "unknown"},
    "near_median": {"polygon": [[185, 195], [320, 205], [1180, 515], [705, 475]],
                    "direction": [0.92, 0.39], "allowed_manoeuvres": "unknown"},
}
path.write_text(json.dumps(scene, indent=2) + "\n")
