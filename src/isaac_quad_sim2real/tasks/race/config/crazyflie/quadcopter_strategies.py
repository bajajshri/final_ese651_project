# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg
        #last actions buffer for reward calculation
        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_distance_to_gate = torch.ones(self.num_envs, device=self.device) * 6.0
        self._last_distance_x_to_gate = torch.ones(self.num_envs, device=self.device) * 6.0
        self._last_distance_x_to_prev_gate = torch.ones(self.num_envs, device=self.device) * 6.0
        self._last_distance_to_desired = torch.ones(self.num_envs, device=self.device) * 6.0
        self.offset_penalty = torch.ones(self.num_envs, device=self.device)
        self._prev_gate_idx = torch.zeros(self.num_envs, device=self.device, dtype=self.env._idx_wp.dtype)
        self._prev_gate_reversed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_gate_reversed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # Compute Euclidean distance in GATE frame (not drone body frame)
        # Current gate before update
        actual_prev_gate_idx = self._prev_gate_idx.clone()
        current_gate_idx = self.env._idx_wp.clone()

        # gate passed reward 
        dist_x_to_gate = self.env._pose_drone_wrt_gate[:, 0]

        gate_passed = ((dist_x_to_gate < 0.0) & (self._last_distance_x_to_gate >= 0.0) & 
            (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < 0.4) & (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < 0.4))
        # gate reversed reward 
        gate_reversed = ((dist_x_to_gate >= 0.0) & (self._last_distance_x_to_gate < 0.0) & 
            (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < 2.0) & (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < 0.7))

        # Gate frame distance (for normal approach)
        dist_gate_frame = torch.abs(self.env._pose_drone_wrt_gate[:, 0])

        dist_to_desired =  dist_gate_frame

        delta_dist_to_desired = dist_to_desired - self._last_distance_x_to_gate
        progress = -delta_dist_to_desired
        self._last_distance_x_to_gate = dist_to_desired.clone()

        #------switch to the next waypoint if gate passed-----
        ids_gate_passed = torch.where(gate_passed)[0]
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        
        # set desired positions in the world frame
        self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
        
        # Store current gate as previous for NEXT timestep
        self._prev_gate_idx = current_gate_idx

        #gate_dir_gate_frame = -self.env._pose_drone_wrt_gate[:, :3]
        #gate_dir_norm = torch.nn.functional.normalize(gate_dir_gate_frame, dim=1)
        #alignment = gate_dir_norm[:, 0].clamp(-1.0, 1.0)

        ang_vel_b = self.env._robot.data.root_ang_vel_b                  
        ang_rate_penalty = torch.linalg.norm(ang_vel_b, dim=1)

        if len(ids_gate_passed) > 0:

            new_pose, _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
            self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
            self.env._robot.data.root_link_pos_w[ids_gate_passed]
            )
            # Newly assigned gate — drone is on correct side so use gate frame
            self._last_distance_x_to_gate[ids_gate_passed] = new_pose[:, 0]
            progress[ids_gate_passed] = 0.0
            self._prev_gate_idx[ids_gate_passed] = current_gate_idx[ids_gate_passed]
        # Compute drone position relative to PREVIOUS gate
        
        prev_gate_pos_w = self.env._waypoints[actual_prev_gate_idx, :3]
        drone_pos_wrt_prev_gate, _ = subtract_frame_transforms(
            prev_gate_pos_w,
            self.env._waypoints_quat[actual_prev_gate_idx, :],
            self.env._robot.data.root_link_pos_w
        )

        # distance to previous gate
        dist_x_to_prev_gate = drone_pos_wrt_prev_gate[:, 0]
        #prev_gate_reversed = ((dist_x_to_prev_gate >= 0.0) & (self._last_distance_x_to_prev_gate < 0.0) & 
        #       (torch.abs(drone_pos_wrt_prev_gate[:, 1]) < 1.0) & (torch.abs(drone_pos_wrt_prev_gate[:, 2]) < 0.6))
        #self._last_distance_x_to_prev_gate = dist_x_to_prev_gate.clone()
        #self.env._prev_gate_reversed = prev_gate_reversed

        # add cost for action change to encourage smoother control (optional)
        control_smoothness_cost = torch.linalg.norm(self.env._actions - self._last_actions, dim=1)
        self._last_actions = self.env._actions.clone()
        control_cost = torch.linalg.norm(self.env._actions, dim=1)

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "gate_passed": gate_passed.float() * self.env.rew['gate_passed_reward_scale'],
                "gate_reversed": gate_reversed.float() * self.env.rew['gate_reversed_reward_scale'],
                #"prev_gate_reversed": prev_gate_reversed.float() * self.env.rew['prev_gate_reversed_reward_scale'],
                "ang_rate":ang_rate_penalty * self.env.rew['ang_rate_reward_scale'],
                "control": control_cost * self.env.rew['control_reward_scale'],
                "control_smoothness": control_smoothness_cost * self.env.rew['control_smoothness_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including waypoint positions and drone state."""
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = torch.bmm(self.env._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self.env._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3)
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3)
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self.env._robot.data.root_com_lin_vel_b,			# 3 dim (linear vel in body frame)
                attitude_mat.view(attitude_mat.shape[0], -1),			# 9 dim (drone rotation matrix)
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),	# 12 dim (corners of current gate)
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),	# 12 dim (corners of next gate)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # Update yaw tracking
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps

        self.env._previous_actions = self.env._actions.clone()

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        num_waypoints = self.env._waypoints.shape[0]
        waypoint_indices = torch.randint(0, num_waypoints, (n_reset,), 
                                 device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, 0.7)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 =  torch.empty(1, device=self.device).uniform_(0.05, 2.0)

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0

        dist_to_initial_gate = torch.linalg.norm(
        self.env._pose_drone_wrt_gate[env_ids], dim=1
        )
        self._last_distance_to_gate[env_ids] = dist_to_initial_gate
        self._last_distance_x_to_prev_gate[env_ids] = 10.0
        self._last_distance_x_to_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self._last_actions[env_ids] = 0.0
        self.offset_penalty[env_ids] = 1.0
        self._prev_gate_idx[env_ids] = waypoint_indices

        self._last_distance_to_desired[env_ids] = torch.abs(
            self.env._pose_drone_wrt_gate[env_ids, 0]
        )
