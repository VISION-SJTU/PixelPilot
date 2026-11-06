# SPDX-License-Identifier: Apache-2.0
"""Lift composite-image predictions to the current sample's LIDAR_TOP frame."""

import argparse

from .formats import parse_waypoints
from .geometry import lift_pixels
from .io import read_records, write_json


def sensor_to_global(nusc, data_token):
    import numpy as np
    from pyquaternion import Quaternion

    sample_data = nusc.get("sample_data", data_token)
    sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(sensor["rotation"]).rotation_matrix
    transform[:3, 3] = sensor["translation"]
    world = np.eye(4)
    world[:3, :3] = Quaternion(pose["rotation"]).rotation_matrix
    world[:3, 3] = pose["translation"]
    return world @ transform


def inverse_projection(nusc, sample, camera):
    import numpy as np

    token = sample["data"][camera]
    data = nusc.get("sample_data", token)
    calibration = nusc.get("calibrated_sensor", data["calibrated_sensor_token"])
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = calibration["camera_intrinsic"]
    lidar_to_camera = np.linalg.inv(sensor_to_global(nusc, token)) @ sensor_to_global(nusc, sample["data"]["LIDAR_TOP"])
    return np.linalg.inv(intrinsics @ lidar_to_camera), data["width"], data["height"]


def unmap_pixel(point, view_width=784, view_height=448):
    """Undo the composite's 180-degree rear rotation in pixel-center coordinates."""
    u, v = point
    if not 0 <= u < view_width or not 0 <= v < 2 * view_height:
        raise ValueError(f"Pixel outside {view_width}x{2 * view_height} composite: {point}")
    if v < view_height:
        return "CAM_FRONT", [u, v]
    return "CAM_BACK", [view_width - 1 - u, 2 * view_height - 1 - v]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--nusc-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--ground-z", type=float, required=True, help="Ground-plane z in current LIDAR_TOP coordinates")
    parser.add_argument("--frame", choices=["LIDAR_TOP", "PLANNING"], default="LIDAR_TOP",
                        help="PLANNING uses [lidar_y, -lidar_x], matching recovered gt_planning labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    records = read_records(args.predictions)
    results = []
    ids = set()
    for row in records:
        token = row.get("id")
        if not isinstance(token, str) or token in ids:
            raise ValueError("Lifting requires unique sample ids on every prediction")
        ids.add(token)
        sample = nusc.get("sample", token)
        matrices = {}
        points = []
        for point in parse_waypoints(row["predict"]):
            if isinstance(point, dict):
                # Historical out-of-view token: a longitudinal LIDAR_TOP coordinate.
                points.append([0.0, point["traj"], args.ground_z])
                continue
            camera, pixel = unmap_pixel(point)
            if camera not in matrices:
                matrices[camera] = inverse_projection(nusc, sample, camera)
            matrix, width, height = matrices[camera]
            native = [[pixel[0] * width / 784, pixel[1] * height / 448]]
            points.append(lift_pixels(native, matrix, args.ground_z)[0].tolist())
        trajectory = [[p[1], -p[0]] if args.frame == "PLANNING" else p[:2] for p in points]
        results.append({"id": token, "trajectory": trajectory, "frame": args.frame, "ground_z": args.ground_z})
    write_json(args.output, results)
    print(f"Lifted {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
