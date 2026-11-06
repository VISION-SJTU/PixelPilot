# Modified for PixelPilot source release (2026-11-05): defer dataset loading, configure local data root, cache calibration; preserve scoring formula.
# SPDX-License-Identifier: Apache-2.0
"""Historical nuScenes reward; scientific limitations are documented in docs/reproduction.md.

Release preparation changes initialization and portability, retaining the scoring formula.
"""
import math
import os
import re
from functools import lru_cache

import numpy as np

@lru_cache(maxsize=1)
def get_nuscenes():
    from nuscenes.nuscenes import NuScenes

    return NuScenes(
        version=os.environ.get("NUSCENES_VERSION", "v1.0-trainval"),
        dataroot=os.environ.get("NUSCENES_ROOT", "data/nuscenes"),
        verbose=False,
    )

def nuscenes_format_reward(predict_str: str) -> float:
    pattern = r"<perception>(.*?)</perception>\s*<action>(.*?)</action>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0

    return thinking_format_reward

def obtain_sensor2top(sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='camera'):
        """
        获取从传感器到LIDAR顶部的变换矩阵

        Args:
            sensor_token: 传感器token
            l2e_t: LIDAR到ego的平移
            l2e_r_mat: LIDAR到ego的旋转矩阵
            e2g_t: ego到全局的平移
            e2g_r_mat: ego到全局的旋转矩阵
            sensor_type: 传感器类型

        Returns:
            sensor2lidar_rotation: 传感器到LIDAR的旋转矩阵
            sensor2lidar_translation: 传感器到LIDAR的平移向量
        """
        from pyquaternion import Quaternion

        nusc = get_nuscenes()
        sd_rec = nusc.get('sample_data', sensor_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        l2e_r_s = cs_record['rotation']
        l2e_t_s = cs_record['translation']
        e2g_r_s = pose_record['rotation']
        e2g_t_s = pose_record['translation']

        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                      ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

        sensor2lidar_rotation = R.T
        sensor2lidar_translation = T

        return sensor2lidar_rotation, sensor2lidar_translation

def project_2d_to_3d(points_2d, img2lidar_rt):
    x, y = points_2d.split(', ')
    x = float(x) * 1600 / 784
    y = float(y) * 900 / 448
    point = np.array([x, y])
    point_hom = np.array([point[0], point[1], 1.0, 1.0])
    point_3d_hom = img2lidar_rt @ point_hom

    if abs(point_3d_hom[3]) > 1e-8:
        point_3d_hom = point_3d_hom / point_3d_hom[3]
    camera_center = img2lidar_rt @ np.array([0, 0, 0, 1])
    if abs(camera_center[3]) > 1e-8:
        camera_center = camera_center / camera_center[3]
    ray_dir = point_3d_hom[:3] - camera_center[:3]
    if abs(ray_dir[2]) > 1e-8:
        t = (-1.0 - camera_center[2]) / ray_dir[2]
        if t > 0:
            point_3d = camera_center[:3] + t * ray_dir
        else:
            point_3d = np.array([0., 0., 0])
    else:
        point_3d = np.array([0., 0., -1.0])

    return point_3d

@lru_cache(maxsize=4096)
def obtain_sensor(id):
    from pyquaternion import Quaternion

    nusc = get_nuscenes()
    sample_token = id
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd_record = nusc.get('sample_data', lidar_token)
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
    lidar_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
    l2e_r = lidar_cs_record['rotation']
    l2e_t = lidar_cs_record['translation']
    e2g_r = lidar_pose_record['rotation']
    e2g_t = lidar_pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    cam_sd_record = nusc.get('sample_data', sample_data_token)
    cam_cs_record = nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
    camera_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
    s2l_r, s2l_t = obtain_sensor2top(sample_data_token,
                                        l2e_t,
                                        l2e_r_mat,
                                        e2g_t,
                                        e2g_r_mat,
                                        'CAM_FRONT')
    lidar2cam_r = np.linalg.inv(s2l_r)
    lidar2cam_t = s2l_t @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T
    img2lidar_rt = np.linalg.inv(lidar2img_rt)

    return img2lidar_rt

def nuscenes_accuracy_reward(predict_str: str, ground_truth: str, id: str) -> float:
    max_accuracy_reward = 0.0

    try:
        answer_match_pred = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if answer_match_pred:
            traj_str_pred = answer_match_pred.group(1)
            trajectory_pattern = r'<[^>]+>|\d+, \d+'
            points_pred = re.findall(trajectory_pattern, traj_str_pred)
            points_gt = re.findall(trajectory_pattern, ground_truth)

            if len(points_pred) == 6:
                l2_3d_reward = 0.0
                l1_2d_reward = 0.0
                traj_format_reward = 0.0
                num_2d = 0

                img2lidar_rt = obtain_sensor(id)

                for i in range(len(points_pred)):
                    traj_in_pred = False
                    traj_in_gt = False
                    if 'traj' in points_pred[i]:
                        traj_in_pred = True
                        match = re.search(r'-?\d+\.\d+', points_pred[i])
                        number = float(match.group(0))
                        pred_3d = np.array([0., number, -1.0])
                    else:
                        pred_3d = project_2d_to_3d(points_pred[i], img2lidar_rt)

                    if 'traj' in points_gt[i]:
                        traj_in_gt = True
                        match = re.search(r'-?\d+\.\d+', points_gt[i])
                        number = float(match.group(0))
                        gt_3d = np.array([0., number, -1.0])
                    else:
                        gt_3d = project_2d_to_3d(points_gt[i], img2lidar_rt)

                    l2_distance = ((pred_3d[0] - gt_3d[0]) ** 2 + (pred_3d[1] - gt_3d[1]) ** 2) ** 0.5
                    l1_distance_x = abs(pred_3d[0] - gt_3d[0])
                    l1_distance_y = abs(pred_3d[1] - gt_3d[1])

                    if not traj_in_pred and not traj_in_gt:
                        x_pred_2d, y_pred_2d = points_pred[i].split(', ')
                        x_pred_2d = float(x_pred_2d)
                        y_pred_2d = float(y_pred_2d)

                        x_gt_2d, y_gt_2d = points_gt[i].split(', ')
                        x_gt_2d = float(x_gt_2d)
                        y_gt_2d = float(y_gt_2d)
                        l1_distance_x_2d = abs(x_pred_2d - x_gt_2d)
                        l1_distance_y_2d = abs(y_pred_2d - y_gt_2d)
                        if l1_distance_x_2d + l1_distance_y_2d < 10:
                            l1_2d_reward += 1
                            num_2d += 1

                    l2_3d_reward += 2 * math.exp(-l2_distance) / (1 + math.exp(-l2_distance))

                    cur_traj_format_reward = 1.0 if traj_in_pred == traj_in_gt else 0.0
                    traj_format_reward += cur_traj_format_reward

                l2_3d_reward = l2_3d_reward / 6
                l1_2d_reward = l1_2d_reward / num_2d if num_2d > 0 else 0.0
                traj_format_reward = traj_format_reward / 6
                max_accuracy_reward = l2_3d_reward + traj_format_reward + l1_2d_reward
    except Exception as e:
        print(f"Error in accuracy reward calculation: {e}")
        print(f"Predict_str: {predict_str}")
        print(f"Ground_truth: {ground_truth}")
        print(f"ID: {id}")

    return max_accuracy_reward

def nuscenes_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('. ')

        sentences = [s.strip() for s in sentences if s.strip()]

        seen = set()
        repeats = 0

        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)

    except Exception:
        pass

    return non_repeat_reward

def nuscenes_action_reward(predict_str: str, ground_truth: str, complexity_weights=None) -> float:
    if complexity_weights is None:
        complexity_weights = {
            "ACCELERATE": 0.9, "DECELERATE": 1.0, "STOP": 1.0, "KEEP": 0.8,
            "FORWARD": 0.8, "LEFT": 1, "RIGHT": 1,
        }
    action_reward = 0.0
    try:
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', predict_str, re.DOTALL)
        if not action_match:
            return action_reward
        action_str = action_match.group(1).strip()
        action_words = set(action_str.split(', '))
        action_words = {word for word in action_words if word in complexity_weights}
        gt_words = set(ground_truth.strip().split(', '))
        gt_words = {word for word in gt_words if word in complexity_weights}
        true_positives = action_words.intersection(gt_words)
        false_positives = action_words - gt_words
        false_negatives = gt_words - action_words
        precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-8)
        recall = len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-8)
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        action_reward = f1_score

    except Exception:
        pass

    return action_reward

def nuscenes_compute_score(predict_str: str, ground_truth: str, id: str) -> float:
    get_nuscenes()
    format_reward = nuscenes_format_reward(predict_str)
    accuracy_reward = nuscenes_accuracy_reward(predict_str, ground_truth, id)
    non_repeat_reward = nuscenes_non_repeat_reward(predict_str)

    reward = format_reward + accuracy_reward + non_repeat_reward # + action_reward
    return reward


