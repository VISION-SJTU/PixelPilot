# SPDX-License-Identifier: Apache-2.0
# Adapted for PixelPilot release from fb_mix_percep_action_traj.py.
# Projection equations retained; injected NuScenes handle replaces hardcoded initialization.
import numpy as np
from pyquaternion import Quaternion

class TrajectoryProjector:

    def __init__(self, nusc):
        self.nusc = nusc

    def obtain_sensor2top(self, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='camera'):
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
        sd_rec = self.nusc.get('sample_data', sensor_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        l2e_r_s = cs_record['rotation']
        l2e_t_s = cs_record['translation']
        e2g_r_s = pose_record['rotation']
        e2g_t_s = pose_record['translation']
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = l2e_r_s_mat.T @ e2g_r_s_mat.T @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        sensor2lidar_rotation = R.T
        sensor2lidar_translation = T
        return (sensor2lidar_rotation, sensor2lidar_translation)

    def preprocess_trajectory(self, plan_traj, start_offset=False):
        """
        预处理轨迹数据
        
        Args:
            plan_traj: 原始轨迹数据 (相对偏移)
            start_offset: 起始点偏移距离
            
        Returns:
            processed_traj: 处理后的轨迹点 (齐次坐标)
        """
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj_homo = np.concatenate((-1.0 * plan_traj[:, [1]], plan_traj[:, [0]], -1.0 * np.ones((plan_traj.shape[0], 1)), np.ones((plan_traj.shape[0], 1))), axis=1)
        return plan_traj_homo

    def project_trajectory_to_camera(self, trajectory_3d, sample_token, camera_name='CAM_FRONT'):
        """
        将3D轨迹投影到相机图像平面
        
        Args:
            trajectory_3d: 3D轨迹点 (齐次坐标)
            sample_token: 样本token
            camera_name: 相机名称
            
        Returns:
            trajectory_2d: 投影后的2D轨迹点
            valid_mask: 有效点掩码
        """
        sample = self.nusc.get('sample', sample_token)
        sample_data_token = sample['data'][camera_name]
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd_record = self.nusc.get('sample_data', lidar_token)
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        lidar_pose_record = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
        l2e_r = lidar_cs_record['rotation']
        l2e_t = lidar_cs_record['translation']
        e2g_r = lidar_pose_record['rotation']
        e2g_t = lidar_pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        cam_sd_record = self.nusc.get('sample_data', sample_data_token)
        cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
        camera_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
        (s2l_r, s2l_t) = self.obtain_sensor2top(sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, camera_name)
        lidar2cam_r = np.linalg.inv(s2l_r)
        lidar2cam_t = s2l_t @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        projected = lidar2img_rt @ trajectory_3d.T
        z_vals = projected[2:3, ...]
        valid_mask = z_vals > 1e-05
        trajectory_2d = projected[0:2, ...] / np.maximum(z_vals, np.ones_like(z_vals) * 1e-05)
        trajectory_2d = trajectory_2d.T
        (img_w, img_h) = (cam_sd_record['width'], cam_sd_record['height'])
        in_bounds_mask = (trajectory_2d[:, 0] >= 0) & (trajectory_2d[:, 0] < img_w) & (trajectory_2d[:, 1] >= 0) & (trajectory_2d[:, 1] < img_h)
        valid_mask = valid_mask.flatten() & in_bounds_mask
        return (trajectory_2d, valid_mask)
