# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import re
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import math
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # 扩展命令维度，新增一个维度用于控制站立/手倒立转换
        cfg.commands.num_commands = 4  # 从4增加到5 (lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_handstand)
        cfg.env.num_observations = 46  # 从45增加到46以适应新的命令维度
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
         # ✅ 新增：Env0 的 episode 计数器
        self.env_episode_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._parse_cfg(self.cfg)
        

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self._debug_print_env0()
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        """确保接触力数据正确刷新"""
        # 刷新所有必要的张量

        

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

                # 检查接触力是否有效
        if self.common_step_counter % 200 == 0:
            total_contact = torch.sum(torch.norm(self.contact_forces, dim=-1)).item()
            print(f"总接触力检查: {total_contact:.6f}")


        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            
        # 确保这些更新在最后执行
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)
        # ✅ 新增：按 episode 编号交替模式
        self.env_episode_count[env_ids] += 1  # 每 reset 一次，episode 数 +1
        is_handstand = (self.env_episode_count[env_ids] % 2 == 1)  # 第1,3,5...轮是 handstand

        # 假设 commands[:, 3] 是手倒立开关（1=handstand, 0=standing）
        self.commands[env_ids, 3] = is_handstand.float()

        # 如果你还有其他命令维度（如前进速度），可以同时清零或设默认值
        self.commands[env_ids, :3] = 0.0  # 例如：lin_vel_x, lin_vel_y, ang_vel_yaw 全为0

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

             # 在重置部分添加渐进控制重置
        if len(env_ids) > 0:
            # 重置渐进控制变量
            self.transition_progress[env_ids] = 0.0
            self.transition_times[env_ids] = torch_rand_float(3.0, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
            self.target_gravity_vec[env_ids] = torch.tensor([0., 0., -1.], device=self.device)
            # 重置站立/手倒立状态
            self.stand_handstand_state[env_ids] = 0  # 默认为站立状态
            self.transition_in_progress[env_ids] = False
    
    def _update_progressive_targets(self):
        """更新渐进控制目标姿态"""
        dt = self.dt
        progress = self.transition_progress
        
        # 使用更平缓的S曲线
        smooth_progress = 3 * progress**2 - 2 * progress**3
        
        # 根据进度调整过渡速度
        speed_factor = 1.0 + 2.0 * (smooth_progress - 0.5)**2
        actual_progress = smooth_progress + self.transition_speed * speed_factor * dt / self.transition_times
        
        self.transition_progress = torch.clamp(actual_progress, 0.0, 1.0)
        
        # 定义姿态序列：水平 → 45度倾斜 → 竖直
        stand_gravity = torch.tensor([0., 0., -1.], device=self.device)
        intermediate_gravity = torch.tensor([0.7, 0., 0.7], device=self.device)  # 45度倾斜
        handstand_gravity = torch.tensor([-1., 0., 0.], device=self.device)
        
        # 使用向量化操作替代if语句
        # 创建掩码：哪些环境处于第一阶段（进度<=0.5）
        stage1_mask = self.transition_progress <= 0.5
        stage2_mask = ~stage1_mask  # 哪些环境处于第二阶段（进度>0.5）
        
        # 第一阶段：水平到45度倾斜
        stage1_progress = self.transition_progress[stage1_mask] * 2  # 映射到[0,1]
        if len(stage1_progress) > 0:
            stage1_target = stand_gravity + stage1_progress.unsqueeze(1) * (intermediate_gravity - stand_gravity)
            self.target_gravity_vec[stage1_mask] = stage1_target
        
        # 第二阶段：45度倾斜到竖直
        stage2_progress = (self.transition_progress[stage2_mask] - 0.5) * 2  # 映射到[0,1]
        if len(stage2_progress) > 0:
            stage2_target = intermediate_gravity + stage2_progress.unsqueeze(1) * (handstand_gravity - intermediate_gravity)
            self.target_gravity_vec[stage2_mask] = stage2_target
   
    def compute_reward(self):
        self.rew_buf[:] = 0.

        is_handstand = (self.commands[:, 3] > 0.5)

        # 计算各项 reward（保留原始值用于 logging）
        standing_rew = self._reward_standing()
        handstand_rew = self._reward_handstand()
        low_torques_rew = self._reward_low_torques()
        handstand_feet_air_time_rew = self._reward_handstand_feet_air_time()
        dof_acc_rew = self._reward_dof_acc()
        dof_vel_rew = self._reward_dof_vel()
        dof_pos_limits_rew = self._reward_dof_pos_limits()
        torques_rew = self._reward_torques()
        torque_smoothness_rew = self._reward_torque_smoothness()
        action_rate_rew = self._reward_action_rate()
        collision_rew = self._reward_collision()
        termination_rew = self._reward_termination() if "termination" in self.reward_scales else 0

        # 构建 reward 字典（用于 logging）
        reward_components = {
            "standing": standing_rew * (~is_handstand),
            "handstand": handstand_rew * is_handstand,
            "handstand_feet_air_time": handstand_feet_air_time_rew * is_handstand,
            "torques": torques_rew,
            # "torque_smoothness": torque_smoothness_rew,
            "action_rate": action_rate_rew,
            "dof_acc": dof_acc_rew,
            "dof_vel" : dof_vel_rew,
            "dof_pos_limits":dof_pos_limits_rew,
            "low_torques": low_torques_rew,
            "collision": collision_rew,
            "termination": termination_rew,
        }

        # 累加到总 reward
        for name, rew in reward_components.items():
            scale = self.reward_scales.get(name, 0.0)
            self.rew_buf += rew * scale

        # ✅ 关键：全部加入 episode_sums（用于日志）
        for name, rew in reward_components.items():
            if name in self.episode_sums:
                self.episode_sums[name] += rew  # 注意：这里加的是 unscaled 原始 reward！

        # ==================== 🐾 调试打印：关键状态观测 ====================
        if self.common_step_counter % 20 == 0:  # 每100步打印一次 env0 的状态
            env_id = 0  # 观察第0个环境
            
            # 获取刚体位置 [num_envs, num_bodies, 3]
            rb_pos = self.rigid_body_pos[env_id]  # shape: [num_bodies, 3]

            # === 根据你的索引定义 ===
            front_feet_idx = [4, 8]      # FL_FOOT, FR_FOOT
            hind_feet_idx = [12, 16]     # HL_FOOT, HR_FOOT
            hind_knee_idx = [11, 15]     # 假设是后膝盖/小腿

            # 提取 z 高度
            front_z = rb_pos[front_feet_idx, 2].cpu().numpy()
            hind_z = rb_pos[hind_feet_idx, 2].cpu().numpy()
            knee_z = rb_pos[hind_knee_idx, 2].cpu().numpy()
            
            base_height = self.root_states[env_id, 2].item()  # base z
            proj_grav = self.projected_gravity[env_id].cpu().numpy()

            mode = "handstand" if is_handstand[env_id] else "standing"

            front_rel_x = rb_pos[front_feet_idx, 0] - self.root_states[0, 0].item()
            mode = "handstand" if self.commands[0, 3] > 0.5 else "standing"
            

            print(f"\n{'='*60}")
            print(f"📊 Env0 Debug | Step {self.common_step_counter} | Mode: {mode}")
            print(f"📊 Env0 Episode #: {self.env_episode_count[0].item()} | Mode: {mode}")
            print(f"   Base height: {base_height:.3f} m")
            print(f"   Projected gravity: [{proj_grav[0]: .2f}, {proj_grav[1]: .2f}, {proj_grav[2]: .2f}]")
            print(f"   Front feet z: [{front_z[0]:.3f}, {front_z[1]:.3f}] m")
            print(f"   Front feet rel x: [{front_rel_x[0]:.3f}, {front_rel_x[1]:.3f}] m")
            print(f"   Hind feet z:  [{hind_z[0]:.3f}, {hind_z[1]:.3f}] m")
            print(f"   Hind knee z:  [{knee_z[0]:.3f}, {knee_z[1]:.3f}] m")
            print(f"   Reward buf[0]: {self.rew_buf[env_id].item():.4f}")
            print(f"   Standing rew: {standing_rew[env_id].item():.4f}")
            print(f"   Handstand rew: {handstand_rew[env_id].item():.4f}")
            print(f"{'='*60}\n")
        # ==============================================================

    # def compute_reward(self):
    #     """ Compute rewards
    #         Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
    #         adds each terms to the episode sums and to the total reward
    #     """
    #      # 在计算奖励前更新渐进控制目标
    #     self._update_progressive_targets()
        
    #     # 根据命令更新站立/手倒立状态
    #     self._update_stand_handstand_state()
        
    #     self.rew_buf[:] = 0.
    #     for i in range(len(self.reward_functions)):
    #         name = self.reward_names[i]
    #         rew = self.reward_functions[i]() * self.reward_scales[name]
    #         self.rew_buf += rew
    #         self.episode_sums[name] += rew
    #     if self.cfg.rewards.only_positive_rewards:
    #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    #     # add termination reward after clipping
    #     if "termination" in self.reward_scales:
    #         rew = self._reward_termination() * self.reward_scales["termination"]
    #         self.rew_buf += rew
    #         self.episode_sums["termination"] += rew
    
    def _update_stand_handstand_state(self):
        """根据命令更新站立/手倒立状态"""
        # 检查命令维度是否包含新的状态转换命令
        if self.commands.shape[1] >= 4:
            # 获取命令中的站立/手倒立转换标志 (第4维，索引为4)
            command_state = self.commands[:, 3]
            
            # 将命令值转换为整数状态 (0或1)
            command_state_int = torch.round(command_state).long()
            
            # 更新状态：0为站立，1为手倒立
            self.stand_handstand_state = command_state_int
            
            # 检查是否需要开始转换
            should_start_transition = (command_state_int != self.current_stand_handstand_state)
            
            # 标记正在进行转换的环境
            self.transition_in_progress = should_start_transition
            
            # 更新当前状态
            self.current_stand_handstand_state = command_state_int.clone()
    
    def compute_observations(self):
        """ Computes observations
        """
          # 使用目标姿态信息替换现有的部分观测（例如替换commands部分）
        target_gravity_obs = self.target_gravity_vec * self.obs_scales.lin_vel

        self.obs_buf = torch.cat((  
            # self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    # target_gravity_obs,  # 用目标姿态替换原来的commands部分
                                    self.commands[:, :4] * self.commands_scale,  # 包含新的命令维度
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind 


        #尝试删除地形高度测量维度
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)



        # add noise if needed
        if self.add_noise:
            # print(f"！！！obs_buf shape: {self.obs_buf.shape}")  # 应该是 torch.Size([num_envs, 45])
            # print(f"！！！noise_scale_vec shape: {self.noise_scale_vec.shape}")  # 应该是 torch.Size([45])
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ 所有环境严格交替 standing / handstand """
        if len(env_ids) == 0:
            return

        # 1. 更新这些 env 的 episode 计数（每个 reset 一次，+1）
        self.env_episode_count[env_ids] += 1

        # 2. 奇数 episode → handstand (1), 偶数 → standing (0)
        is_handstand = (self.env_episode_count[env_ids] % 2 == 1)

        # 3. 设置命令
        self.commands[env_ids, 3] = is_handstand.float()  # 第4维：stand/handstand

        # 4. 其他命令（速度等）设为0，因为两种模式都不需要移动
        self.commands[env_ids, :3] = 0.0

        # （可选）如果你以后想加 curriculum，可以在这里动态调整

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    # def update_command_curriculum(self, env_ids):
    #     """ Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # If the tracking reward is above 80% of the maximum, increase the range of commands
    #     if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
    #         self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
    #         self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def update_command_curriculum(self, env_ids):
        """
        自定义课程更新：根据手倒立成功率提升命令难度
        """
        if not self.cfg.commands.curriculum:
            return

        # 只有当所有环境都达到一定成功率时，才提升难度
        success_rate = self.compute_handstand_success_rate()
        if success_rate > 0.8:  # 80% 环境成功
            self.command_ranges["lin_vel_x"] = [
                -self.cfg.commands.max_curriculum,
                self.cfg.commands.max_curriculum
            ]
            self.command_ranges["lin_vel_y"] = [
                -self.cfg.commands.max_curriculum,
                self.cfg.commands.max_curriculum
            ]
            self.command_ranges["ang_vel_yaw"] = [
                -self.cfg.commands.max_curriculum,
                self.cfg.commands.max_curriculum
            ]
            # 注意：stand_handstand 命令范围保持 [0,1] 不变

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # 直接使用配置中的观测维度，而不是依赖 self.obs_buf
        obs_dim = self.cfg.env.num_observations
        noise_vec = torch.zeros(obs_dim, device=self.device)  # 直接创建，不依赖 obs_buf
        
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 根据观测结构设置噪声尺度
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:10] = 0. # commands
        noise_vec[10:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[22:34] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[34:46] = 0. # previous actions
        # noise_vec[45] = 0. # 新增的命令维度
        
        # 添加调试信息
        print(f"创建的噪声向量维度: {noise_vec.shape[0]}, 配置观测维度: {obs_dim}")
        
        return noise_vec
    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state=self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states=gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos=self.rigid_body_states.view(self.num_envs,self.num_bodies,13)[...,0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        # self.rigid_body_pos = self.rigid_body_states[..., :3]  # 提取 xyz 坐标
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        # 在初始化 noise_scale_vec 之前添加调试
        print(f"=== 初始化调试 ===")
        print(f"配置中的观测维度: {self.cfg.env.num_observations}")
        
        # 先创建 obs_buf
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_observations, 
                                device=self.device, requires_grad=False)
        print(f"创建的 obs_buf 形状: {self.obs_buf.shape}")
        
        # 然后创建 noise_scale_vec
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        print(f"创建的 noise_scale_vec 形状: {self.noise_scale_vec.shape}")
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([-1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading, stand_handstand
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1.0], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
         # 初始化平滑性奖励相关变量
        self.last_dof_acc = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)

         # 渐进控制相关变量
        self.transition_progress = torch.zeros(self.num_envs, device=self.device)  # 过渡进度 (0:站立 → 1:倒立)
        self.transition_times = torch_rand_float(3.0, 5.0, (self.num_envs, 1), device=self.device).squeeze(1)  # 每个环境的过渡时间(3-6秒)
        self.target_gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)  # 目标重力向量
        self.target_gravity_vec[:] = torch.tensor([0., 0., -1.], device=self.device)  # 初始为站立姿态
        # 添加缺失的 transition_speed 初始化
        self.transition_speed = torch_rand_float(0.5, 1.5, (self.num_envs, 1), device=self.device).squeeze(1)
        
        # 新增：站立/手倒立状态变量
        self.stand_handstand_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0:站立, 1:手倒立
        self.current_stand_handstand_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 当前状态
        self.transition_in_progress = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否正在进行转换
        
        # 调试：打印所有刚体名称
        print("=== 所有刚体名称 ===")
        for i, name in enumerate(self.rigid_body_names):
            print(f"{i}: {name}")
        
        # 检查膝盖匹配
        knee_keywords = ['knee', 'thigh', 'shank', 'calf', 'upper_leg', 'lower_leg']
        knee_indices = []
        for i, name in enumerate(self.rigid_body_names):
            name_lower = name.lower()
            for keyword in knee_keywords:
                if keyword in name_lower:
                    knee_indices.append(i)
                    print(f"检测到膝盖部位: {name} (索引: {i})")
                    break
        
        if not knee_indices:
            print("警告：未检测到任何膝盖部位！")
        else:
            print(f"总共检测到 {len(knee_indices)} 个膝盖部位")


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.rigid_body_names=body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1) 
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    
    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        
    #     # 添加维度检查和调整
    #     if contact.shape != self.last_contacts.shape:
    #         print(f"Warning: Dimension mismatch - contact: {contact.shape}, last_contacts: {self.last_contacts.shape}")
    #         # 自动调整到最小公共维度
    #         min_feet = min(contact.shape[1], self.last_contacts.shape[1])
    #         contact = contact[:, :min_feet]
    #         self.last_contacts = self.last_contacts[:, :min_feet]
    #         # 同时调整 feet_air_time 的维度
    #         self.feet_air_time = self.feet_air_time[:, :min_feet]
        
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # def _reward_handstand_feet_height_exp(self):
    #     """改进版：详细的接触力调试"""
        
    #     # 1. 获取膝盖索引
    #     knee_indices = [2, 3, 6, 7, 10, 11, 14, 15]  # 直接使用索引
    #     knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.device)
        
    #     # 2. 详细检查接触力
    #     knee_contact_forces = torch.norm(self.contact_forces[:, knee_indices_tensor, :], dim=-1)
        
    #     # 3. 调试：打印详细的接触力信息
    #     if self.common_step_counter % 100 == 0:
    #         print(f"\n=== 步骤 {self.common_step_counter} 膝盖接触力调试 ===")
            
    #         # 检查接触力张量是否全为0
    #         total_contact = torch.sum(knee_contact_forces).item()
    #         print(f"膝盖总接触力: {total_contact:.6f}")
            
    #         if total_contact < 0.0001:
    #             print("警告：膝盖接触力似乎全为0！")
    #             print("检查接触力张量刷新时机...")
            
    #         # 检查每个膝盖的最大接触力
    #         max_forces, _ = torch.max(knee_contact_forces, dim=0)
    #         for i, idx in enumerate(knee_indices):
    #             body_name = self.rigid_body_names[idx]
    #             max_force = max_forces[i].item()
    #             print(f"  {body_name}: {max_force:.6f}")
            
    #         # 检查是否有任何接触力超过阈值
    #         threshold = 0.1
    #         above_threshold = knee_contact_forces > threshold
    #         count_above = torch.sum(above_threshold).item()
    #         print(f"超过阈值{threshold}的接触点数量: {count_above}/{knee_contact_forces.numel()}")
        
    #     # 4. 使用非常低的阈值检测接触
    #     contact_threshold = 0.01  # 非常低的阈值
    #     knee_contact = knee_contact_forces > contact_threshold
    #     any_knee_contact = knee_contact.any(dim=1)
        
    #     # 5. 计算脚部高度奖励
    #     feet_indices = [4, 8, 12, 16]  # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)
        
    #     foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
    #     feet_height = foot_pos[..., 2]
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
    #     std = self.cfg.params.handstand_feet_height_exp["std"]
    #     feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
    #     height_reward = torch.exp(-feet_height_error / (std**2))
        
    #     # 6. 应用膝盖接触惩罚
    #     reward = height_reward * (~any_knee_contact).float()
        
    #     # 7. 详细的调试信息
    #     if self.common_step_counter % 100 == 0:
    #         knee_contact_rate = torch.mean(any_knee_contact.float()).item() * 100
    #         avg_reward = torch.mean(reward).item()
    #         avg_height_reward = torch.mean(height_reward).item()
            
    #         print(f"膝盖接触率: {knee_contact_rate:.1f}%")
    #         print(f"高度奖励: {avg_height_reward:.3f}")
    #         print(f"最终奖励: {avg_reward:.3f}")
    #         print(f"接触环境数量: {torch.sum(any_knee_contact).item()}/{self.num_envs}")
            
    #         # 检查奖励是否被正确应用
    #         if knee_contact_rate > 0:
    #             contact_envs = any_knee_contact.nonzero(as_tuple=False).flatten()
    #             if len(contact_envs) > 0:
    #                 env_id = contact_envs[0].item()
    #                 print(f"示例环境 {env_id}: 膝盖接触力 = {knee_contact_forces[env_id]}")
    #                 print(f"示例环境 {env_id}: 高度奖励 = {height_reward[env_id]:.3f}")
    #                 print(f"示例环境 {env_id}: 最终奖励 = {reward[env_id]:.3f}")
    #         print("---")
        
    #     return reward
        
    # def _reward_handstand_feet_height_exp(self):
    #     """优化版：基于0.022米阈值的抬腿判断"""
        
    #     # 1. 获取相关刚体索引
    #     thigh_indices = [2, 6, 10, 14]    # FL_THIGH, FR_THIGH, HL_THIGH, HR_THIGH
    #     shank_indices = [3, 7, 11, 15]    # FL_SHANK, FR_SHANK, HL_SHANK, HR_SHANK
    #     foot_indices = [4, 8, 12, 16]     # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
        
    #     # 2. 计算膝盖离地高度
    #     shank_pos = self.rigid_body_pos[:, shank_indices, :]
    #     knee_heights = shank_pos[..., 2]
        
    #     # 膝盖安全高度阈值
    #     knee_safe_height = 0.05
    #     knee_height_penalty = torch.sum(torch.where(knee_heights < knee_safe_height,
    #                                             (knee_safe_height - knee_heights) ** 2, 0.0), dim=1)
    #     knee_safety_reward = torch.exp(-knee_height_penalty / 0.05)
        
    #     # 3. 前腿脚部高度奖励 - 关键修改：基于0.022米阈值
    #     front_foot_indices = [4, 8]
    #     front_foot_tensor = torch.tensor(front_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     front_foot_pos = self.rigid_body_pos[:, front_foot_tensor, :]
    #     front_foot_height = front_foot_pos[..., 2]
    #     front_foot_x = front_foot_pos[..., 0]
        
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
        
    #     # 定义抬腿阈值
    #     LIFT_THRESHOLD = 0.025  # 高度大于0.022米才算是抬腿
        
    #     # 分别获取左右前腿高度
    #     front_left_height = front_foot_height[:, 0]
    #     front_right_height = front_foot_height[:, 1]
        
    #     # 判断每条腿的状态
    #     left_leg_lifted = front_left_height > LIFT_THRESHOLD  # 左腿是否抬离地面
    #     right_leg_lifted = front_right_height > LIFT_THRESHOLD  # 右腿是否抬离地面
    #     both_legs_lifted = left_leg_lifted & right_leg_lifted  # 双腿都抬离
    #     any_leg_lifted = left_leg_lifted | right_leg_lifted  # 任意腿抬离
        
    #     # 计算实际抬腿高度（只考虑抬离地面的腿）
    #     left_lift_amount = torch.clamp(front_left_height - LIFT_THRESHOLD, 0)
    #     right_lift_amount = torch.clamp(front_right_height - LIFT_THRESHOLD, 0)
    #     total_lift_amount = left_lift_amount + right_lift_amount
        
    #     # 新策略：基于抬腿状态的奖励系统
    #     # 1. 基础抬腿奖励（鼓励至少一条腿抬离地面）
    #     base_lift_reward = any_leg_lifted.float() * 0.3
        
    #     # 2. 单腿抬高奖励（鼓励抬得更高）
    #     single_leg_reward = (
    #         torch.max(left_lift_amount, right_lift_amount) / (target_height - LIFT_THRESHOLD)
    #     ) * 0.4
        
    #     # 3. 双腿协调奖励（鼓励双腿都抬离地面）
    #     both_legs_reward = both_legs_lifted.float() * 0.5
    #     min_lift_reward = (
    #         torch.min(left_lift_amount, right_lift_amount) / (target_height - LIFT_THRESHOLD)
    #     ) * 0.3
        
    #     # 4. 交替模式特别奖励（一条腿抬高，一条腿支撑）
    #     alternation_condition = (left_leg_lifted & ~right_leg_lifted) | (~left_leg_lifted & right_leg_lifted)
    #     alternation_reward = alternation_condition.float() * 0.4
        
    #     # 5. 目标高度奖励（针对已抬离的腿）
    #     lifted_heights = torch.where(any_leg_lifted.unsqueeze(1), front_foot_height, torch.tensor(LIFT_THRESHOLD, device=self.device))
    #     height_error = torch.sum((lifted_heights - target_height) ** 2, dim=1)
    #     target_reward = torch.exp(-height_error / 0.3) * 0.6
        
    #     # 组合高度奖励
    #     height_reward = (
    #         base_lift_reward +
    #         single_leg_reward + 
    #         both_legs_reward +
    #         min_lift_reward +
    #         alternation_reward +
    #         target_reward
    #     )
        
    #     # 4. 抬腿不足惩罚（针对应该抬腿但没抬的情况）
    #     # 如果机器人的最大高度已经超过阈值，但某条腿还在地上，给予惩罚
    #     if hasattr(self, 'max_achieved_height'):
    #         should_lift = self.max_achieved_height > LIFT_THRESHOLD * 2  # 如果曾经达到较高高度
    #         lift_penalty = torch.where(
    #             should_lift & ~both_legs_lifted,
    #             (1.0 - both_legs_lifted.float()) * 0.2,  # 惩罚没抬腿的情况
    #             0.0
    #         )
    #     else:
    #         lift_penalty = torch.zeros(self.num_envs, device=self.device)
    #         self.max_achieved_height = torch.max(front_foot_height, dim=1)[0]
        
    #     # 更新最大高度
    #     self.max_achieved_height = torch.max(self.max_achieved_height, torch.max(front_foot_height, dim=1)[0])
        
    #     # 5. 前腿向后伸展惩罚
    #     backward_penalty_threshold = 0.0
    #     backward_penalty = torch.sum(torch.where(front_foot_x < backward_penalty_threshold,
    #                                         (backward_penalty_threshold - front_foot_x) ** 2, 0.0), dim=1)
    #     backward_penalty_reward = torch.exp(-backward_penalty / 0.1)
        
    #     # 6. 后腿稳定性奖励
    #     hind_foot_indices = [12, 16]
    #     hind_foot_tensor = torch.tensor(hind_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     hind_foot_pos = self.rigid_body_pos[:, hind_foot_tensor, :]
    #     hind_foot_height = hind_foot_pos[..., 2]
    #     hind_target_height = 0.05
    #     hind_height_error = torch.sum((hind_foot_height - hind_target_height) ** 2, dim=1)
    #     hind_reward = torch.exp(-hind_height_error / 0.05)
        
    #     # 7. 组合奖励
    #     combined_reward_before = (
    #         knee_safety_reward * 0.2 +
    #         height_reward * 0.8 +
    #         backward_penalty_reward * 0. +
    #         hind_reward * 0. -
    #         lift_penalty  # 抬腿不足惩罚
    #     )
        
    #     # 8. 强惩罚：膝盖触地
    #     severe_knee_contact = torch.any(knee_heights < 0.05, dim=1)
    #     combined_reward = combined_reward_before.clone()
    #     combined_reward[severe_knee_contact] = 0.0
        
    #     # 9. 根据站立/手倒立命令调整奖励
    #     # 如果命令是站立(0)，但脚部高度过高，应给予惩罚
    #     standing_command = (self.stand_handstand_state == 0)
    #     handstand_command = (self.stand_handstand_state == 1)
        
    #     # 站立命令时，脚部应接近地面
    #     standing_penalty = standing_command.float() * (
    #         torch.exp(-torch.min(front_foot_height, dim=1)[0] / 0.1)  # 鼓励脚部接近地面
    #     )
        
    #     # 手倒立命令时，脚部应抬高
    #     handstand_reward = handstand_command.float() * height_reward
        
    #     # 最终奖励：根据命令选择适当的奖励
    #     final_reward = standing_penalty + handstand_reward
        
    #     # 10. 详细调试信息
    #     if self.common_step_counter % 50 == 0:
    #         for i in range(min(1, severe_knee_contact.shape[0])):  # 只打印第一个环境的调试信息
    #             min_height = knee_heights[i].min().item()
    #             contact = severe_knee_contact[i].item()
    #             reward_before = combined_reward_before[i].item()
    #             reward_after = combined_reward[i].item()
                
    #             # 获取前腿信息
    #             left_height = front_left_height[i].item()
    #             right_height = front_right_height[i].item()
    #             left_lifted = left_leg_lifted[i].item()
    #             right_lifted = right_leg_lifted[i].item()
    #             left_lift_amt = left_lift_amount[i].item()
    #             right_lift_amt = right_lift_amount[i].item()
                
    #             command_state = self.stand_handstand_state[i].item()
                
    #             print(f"环境{i}: 最低膝高={min_height:.3f}, 触地={contact}, 命令状态={command_state}")
    #             print(f"  抬腿状态 (阈值={LIFT_THRESHOLD:.3f}m):")
    #             print(f"    - 左腿: 高度={left_height:.3f}m, 抬腿={left_lifted}, 抬升量={left_lift_amt:.3f}m")
    #             print(f"    - 右腿: 高度={right_height:.3f}m, 抬腿={right_lifted}, 抬升量={right_lift_amt:.3f}m")
    #             print(f"    - 状态: 单腿抬离={any_leg_lifted[i].item()}, 双腿抬离={both_legs_lifted[i].item()}")
                
    #             print(f"  奖励分量:")
    #             print(f"    - 基础抬腿: {base_lift_reward[i].item():.3f}")
    #             print(f"    - 单腿抬高: {single_leg_reward[i].item():.3f}")
    #             print(f"    - 双腿协调: {both_legs_reward[i].item():.3f}")
    #             print(f"    - 最小抬腿: {min_lift_reward[i].item():.3f}")
    #             print(f"    - 交替模式: {alternation_reward[i].item():.3f}")
    #             print(f"    - 目标奖励: {target_reward[i].item():.3f}")
    #             print(f"    - 抬腿惩罚: -{lift_penalty[i].item():.3f}")
    #             print(f"    - 站立惩罚: {standing_penalty[i].item():.3f}")
    #             print(f"    - 倒立奖励: {handstand_reward[i].item():.3f}")
                
    #             print(f"  奖励汇总: 惩罚前={reward_before:.3f}, 惩罚后={reward_after:.3f}, 最终={final_reward[i].item():.3f}")
                
    #             # 给出行为建议
    #             if command_state == 0:
    #                 print(f"  💡 建议: 当前命令为站立，应保持脚部接近地面")
    #             else:
    #                 print(f"  💡 建议: 当前命令为手倒立，应抬起脚部")
                    
    #             print(f"  {'='*60}")

    #     return final_reward
    # def _reward_handstand_feet_height_exp(self):
    #     """基于课程学习的优化版：三阶段手倒立训练奖励函数（含前腿高度对称性约束 + 防止向后摆腿）"""
        
    #     # 1. 获取相关刚体索引
    #     thigh_indices = [2, 6, 10, 14]    # FL_THIGH, FR_THIGH, HL_THIGH, HR_THIGH
    #     shank_indices = [3, 7, 11, 15]    # FL_SHANK, FR_SHANK, HL_SHANK, HR_SHANK
    #     foot_indices = [4, 8, 12, 16]     # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
        
    #     # === 命令状态 ===
    #     handstand_cmd = (self.stand_handstand_state == 1)   # (num_envs,)
    #     standing_cmd = ~handstand_cmd

    #     # 2. 计算膝盖离地高度（用于后续约束）
    #     shank_pos = self.rigid_body_pos[:, shank_indices, :]
    #     knee_heights = shank_pos[..., 2]
        
    #     # 获取当前课程阶段
    #     stage = self.get_current_stage()  # 返回 0, 1, 或 2

    #     # 3. 前腿脚部高度与X位置
    #     front_foot_indices = [4, 8]
    #     front_foot_tensor = torch.tensor(front_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     front_foot_pos = self.rigid_body_pos[:, front_foot_tensor, :]
    #     front_foot_height = front_foot_pos[..., 2]
    #     front_foot_x = front_foot_pos[..., 0]
        
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
    #     symmetry_scale = self.cfg.params.handstand_feet_height_exp.get("symmetry_scale", 0.05)
    #     min_front_x = self.cfg.params.handstand_feet_height_exp.get("min_front_foot_x", -0.25)

    #     # 定义抬腿阈值
    #     LIFT_THRESHOLD = 0.025

    #     # 分别获取左右前腿高度
    #     front_left_height = front_foot_height[:, 0]
    #     front_right_height = front_foot_height[:, 1]
    #     height_diff = torch.abs(front_left_height - front_right_height)
        
    #     left_leg_lifted = front_left_height > LIFT_THRESHOLD
    #     right_leg_lifted = front_right_height > LIFT_THRESHOLD
    #     both_legs_lifted = left_leg_lifted & right_leg_lifted
    #     any_leg_lifted = left_leg_lifted | right_leg_lifted

    #     # 初始化奖励分量
    #     base_lift_reward = torch.zeros_like(front_left_height)
    #     knee_safety_reward = torch.ones_like(front_left_height)
    #     backward_penalty_reward = torch.ones_like(front_left_height)
    #     symmetry_reward_value = torch.ones_like(front_left_height)
    #     hip_forward_reward = torch.ones_like(front_left_height)
    #     target_reward = torch.zeros_like(front_left_height)

    #     if handstand_cmd.any():
    #         mask = handstand_cmd

    #         # —————— 阶段 0：前腿抬起 + 后脚着地 + 防止过度后移 ——————
    #         if stage >= 0:
    #             base_lift_reward[mask] = any_leg_lifted[mask].float() * 1.0
                
    #             # 后脚着地奖励
    #             hind_foot_indices = [12, 16]
    #             hind_foot_tensor = torch.tensor(hind_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #             hind_foot_pos = self.rigid_body_pos[:, hind_foot_tensor, :]
    #             hind_foot_height = hind_foot_pos[..., 2]
    #             hind_on_ground = torch.max(hind_foot_height, dim=1).values < 0.05
    #             hind_on_ground_reward = hind_on_ground[mask].float() * 0.5
    #             base_lift_reward[mask] += hind_on_ground_reward

    #             # === 防止前脚过度后移 ===
    #             excess_backward = torch.relu(min_front_x - front_foot_x)  # (num_envs, 2)
    #             backward_penalty = torch.sum(excess_backward, dim=1)
    #             backward_penalty_reward = torch.clamp(1.0 - backward_penalty / 0.5, min=0.0)

    #             # === 鼓励髋关节前摆 ===
    #             front_hip_indices = [1, 7]  # FL_HIPY, FR_HIPY
    #             front_hip_angles = self.dof_pos[:, front_hip_indices]
    #             target_hip_angle = torch.tensor([0.2, 0.2], device=self.device).unsqueeze(0)
    #             hip_error = torch.sum((front_hip_angles - target_hip_angle) ** 2, dim=1)
    #             hip_forward_reward = torch.exp(-hip_error / 0.1)

    #         # —————— 阶段 1：加入膝盖不触地 + 对称性约束 ——————
    #         if stage >= 1:
    #             hind_knee_indices = [11, 15]
    #             hind_knee_tensor = torch.tensor(hind_knee_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #             hind_knee_pos = self.rigid_body_pos[:, hind_knee_tensor, :]
    #             hind_knee_heights = hind_knee_pos[..., 2]
    #             min_hind_knee = torch.min(hind_knee_heights, dim=1).values
    #             knee_safe = min_hind_knee > 0.05
                
    #             knee_safety_reward[mask] = knee_safe[mask].float() * 0.5 + \
    #                                     torch.exp(-(0.05 - min_hind_knee[mask]).clamp(min=0) / 0.02) * 0.5

    #             symmetry_active = both_legs_lifted[mask]
    #             sym_reward_temp = torch.exp(-height_diff[mask] / symmetry_scale)
    #             symmetry_reward_value[mask] = torch.where(symmetry_active, sym_reward_temp, torch.ones_like(sym_reward_temp))

    #         # —————— 阶段 2：目标高度奖励 ——————
    #         if stage >= 2:
    #             lifted_mask = any_leg_lifted.unsqueeze(1)
    #             effective_height = torch.where(lifted_mask, front_foot_height, torch.full_like(front_foot_height, LIFT_THRESHOLD))
    #             height_error = torch.sum((effective_height - target_height) ** 2, dim=1)
    #             target_reward[mask] = torch.exp(-height_error[mask] / 0.1) * 1.0

    #     # —————— 根据阶段组合奖励 ——————
    #     if stage == 0:
    #         combined_reward = base_lift_reward * 1.0
    #     elif stage == 1:
    #         combined_reward = (
    #             base_lift_reward * 0.2 +
    #             knee_safety_reward * 0.4 +
    #             symmetry_reward_value * 0.1 +
    #             hip_forward_reward * 0.3
    #         )
    #     else:  # stage == 2
    #         combined_reward = (
    #             base_lift_reward * 0.05 +
    #             target_reward * 0.5 +
    #             knee_safety_reward * 0.2  +
    #             symmetry_reward_value * 0.15 +
    #             hip_forward_reward * 0.1
    #         )

    #     # 应用方向性约束（所有阶段）
    #     combined_reward = combined_reward * backward_penalty_reward

    #     # ✅ 关键修复：不再全局硬惩罚膝盖触地！
    #     # 膝盖约束仅通过 stage>=1 的 knee_safety_reward 软性体现
    #     final_reward = combined_reward.clone()

    #     # ========== 手部高度不足惩罚（可保留，但建议提高阈值或移除）==========
    #     hand_lift_threshold = 0.03
    #     hand_lift_insufficient = (front_left_height < hand_lift_threshold) | (front_right_height < hand_lift_threshold)
    #     handstand_hand_failure = handstand_cmd & hand_lift_insufficient
    #     final_reward = torch.where(handstand_hand_failure, torch.zeros_like(final_reward), final_reward)

    #     # ========== 防止前脚甩飞 ==========
    #     front_foot_x_abs_too_large = torch.any(torch.abs(front_foot_x) > 1.0, dim=1)
    #     flyaway_failure = handstand_cmd & front_foot_x_abs_too_large
    #     final_reward = torch.where(flyaway_failure, torch.zeros_like(final_reward), final_reward)

    #     # ========== 站立命令下的奖励（保持不变）==========
    #     standing_penalty = torch.zeros_like(final_reward)
    #     if standing_cmd.any():
    #         min_front_z = torch.min(front_foot_height, dim=1).values
    #         z_reward = torch.exp(-min_front_z / 0.05)

    #         base_x = self.root_states[:, 0]
    #         front_foot_rel_x = front_foot_x - base_x.unsqueeze(1)
    #         max_abs_rel_x = torch.max(torch.abs(front_foot_rel_x), dim=1).values
    #         x_reward = torch.exp(-max_abs_rel_x / 0.1)

    #         standing_penalty = standing_cmd.float() * z_reward * x_reward

    #     # ========== 最终奖励 ==========
    #     raw_final_reward = torch.where(handstand_cmd, final_reward, standing_penalty)

    #     # ========== 课程学习更新 ==========
    #     self.update_curriculum_stage()

    #     # ========== 调试信息（保持不变）==========
    #     if self.common_step_counter % 50 == 0:
    #         i = 0
    #         min_height = knee_heights[i].min().item()
    #         contact = min_height < 0.04  # 仅用于打印
    #         reward_before = combined_reward[i].item()
    #         reward_after = final_reward[i].item()
            
    #         left_height = front_left_height[i].item()
    #         right_height = front_right_height[i].item()
    #         left_x = front_foot_x[i, 0].item()
    #         right_x = front_foot_x[i, 1].item()
    #         height_diff_val = height_diff[i].item()
    #         left_lifted = left_leg_lifted[i].item()
    #         right_lifted = right_leg_lifted[i].item()
            
    #         command_state = self.stand_handstand_state[i].item()
    #         hand_failure = handstand_hand_failure[i].item()
    #         sym_reward_i = symmetry_reward_value[i].item()
    #         hip_reward_i = hip_forward_reward[i].item()
    #         back_penalty_i = backward_penalty_reward[i].item()
    #         flyaway_fail_i = flyaway_failure[i].item()
            
    #         print(f"阶段 {stage} | 环境{i}: 最低膝高={min_height:.3f}, 触地={contact}, 命令状态={command_state}")
    #         print(f"  前脚状态:")
    #         print(f"    - 左: 高度={left_height:.3f}m, X={left_x:.3f}m")
    #         print(f"    - 右: 高度={right_height:.3f}m, X={right_x:.3f}m")
    #         print(f"    - 高度差: {height_diff_val:.3f}m")
    #         print(f"    - 抬腿: 左={left_lifted}, 右={right_lifted}")
    #         print(f"  约束检查:")
    #         print(f"    - 最小X阈值: {min_front_x:.3f}m")
    #         print(f"    - 向后惩罚: {back_penalty_i:.3f}")
    #         print(f"    - 髋关节奖励: {hip_reward_i:.3f}")
    #         print(f"    - 对称奖励: {sym_reward_i:.3f}")
    #         print(f"    - 甩飞失败: {flyaway_fail_i}")
    #         print(f"  奖励: 惩罚前={reward_before:.3f}, 惩罚后={reward_after:.3f}, 最终={raw_final_reward[i].item():.3f}")
            
    #         if command_state == 1 and (left_x < min_front_x or right_x < min_front_x):
    #             print(f"  💡 警告: 前脚过于靠后！应向前伸手（X ≥ {min_front_x:.2f}）")

    #         print(f"  {'='*60}")

    #     return raw_final_reward

    # def get_current_stage(self):
    #     """获取当前课程学习阶段"""
    #     if not hasattr(self, 'curriculum_stage'):
    #         self.curriculum_stage = 0
    #     return self.curriculum_stage


    # def compute_handstand_success_rate(self):
    #     """
    #     根据当前课程阶段动态判断手倒立是否成功：
    #     - 阶段 0: 前脚抬起 + 后脚着地
    #     - 阶段 1/2: 前脚抬起 + 后脚着地 + 后膝离地
    #     """
    #     handstand_envs = (self.stand_handstand_state == 1)
    #     if not handstand_envs.any():
    #         return 0.0

    #     stage = self.get_current_stage()

    #     # 前脚（FL_FOOT=4, FR_FOOT=8）必须抬高 >3cm
    #     front_foot_height = self.rigid_body_pos[:, [4, 8], 2]
    #     front_lifted = (front_foot_height > 0.03).all(dim=1)

    #     # 后脚（HL_FOOT=12, HR_FOOT=16）必须着地 <5cm
    #     hind_foot_height = self.rigid_body_pos[:, [12, 16], 2]
    #     hind_on_ground = (hind_foot_height < 0.022).all(dim=1)

    #     if stage == 0:
    #         # 阶段0：不检查膝盖
    #         success = front_lifted & hind_on_ground
    #     else:
    #         # 阶段1+：后膝（HL_SHANK=11, HR_SHANK=15）必须离地 >5cm
    #         hind_knee_height = self.rigid_body_pos[:, [11, 15], 2]
    #         hind_knee_clear = (hind_knee_height > 0.05).all(dim=1)
    #         success = front_lifted & hind_on_ground & hind_knee_clear

    #     # 仅对手倒立命令环境计数
    #     final_success = success & handstand_envs
    #     return final_success.float().mean().item()

    # def update_curriculum_stage(self):
    #     """安全更新手倒立课程阶段，并定期打印训练进度"""
    #     # 初始化课程状态（仅一次）
    #     if not hasattr(self, '_handstand_curriculum'):
    #         self._handstand_curriculum = {
    #             'stage': 0,
    #             'thresholds': [0.2, 0.5],
    #         }
    #         print("[课程学习] 初始化手倒立课程：阶段 0")

    #     current_stage = self._handstand_curriculum['stage']
    #     if current_stage >= 2:
    #         # 已到最终阶段，但仍可打印进度
    #         if not hasattr(self, '_last_print_step'):
    #             self._last_print_step = 0
    #         if self.common_step_counter - self._last_print_step >= 100:
    #             success_rate = self.compute_handstand_success_rate()
    #             print(f"[课程进度] 🟢 阶段 {current_stage}（已完成）| 当前成功率: {success_rate:.2%}")
    #             self._last_print_step = self.common_step_counter
    #         return

    #     success_rate = self.compute_handstand_success_rate()
    #     thresholds = self._handstand_curriculum['thresholds']

    #     # 定期打印进度（每100步）
    #     if not hasattr(self, '_last_print_step'):
    #         self._last_print_step = 0
    #     if self.common_step_counter - self._last_print_step >= 100:
    #         print(f"[课程进度] 📊 阶段 {current_stage} | 成功率: {success_rate:.2%} "
    #             f"(目标: >{thresholds[current_stage]:.1%})")
    #         self._last_print_step = self.common_step_counter

    #     # 尝试升级阶段
    #     if success_rate > thresholds[current_stage]:
    #         self._handstand_curriculum['stage'] += 1
    #         new_stage = self._handstand_curriculum['stage']
    #         stage_desc = ["前腿抬起 + 后脚着地", "后膝离地", "目标高度"]
    #         print(f"[课程学习] 🎯 进入阶段 {new_stage}: {stage_desc[new_stage]} "
    #             f"(成功率: {success_rate:.2%} > {thresholds[current_stage]:.1%})")
   
    # def get_current_stage(self):
    #     """安全获取当前手倒立课程阶段，首次调用自动初始化"""
    #     if not hasattr(self, '_handstand_curriculum'):
    #         self._handstand_curriculum = {
    #             'stage': 0,
    #             'thresholds': [0.2, 0.5],  # 阶段0→1需60%成功率，阶段1→2需70%
    #         }
    #     return self._handstand_curriculum['stage']


    # def _reward_handstand_feet_height_exp(self):
    #     feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
    #     # print(feet_indices)
    #     # print("Rigid body pos shape:", self.rigid_body_pos.shape)
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
    #     feet_height = foot_pos[..., 2]
    #     # print(feet_height)
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
    #     std = self.cfg.params.handstand_feet_height_exp["std"]
    #     feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
    #     # print(torch.exp(-feet_height_error / (std**2)))
    #     return torch.exp(-feet_height_error / (std**2))
    #     # return 0



    # def _reward_handstand_feet_on_air(self):
    #     """
    #     脚部在空奖励：
    #     1. 使用 self.contact_forces 判断足部是否接触地面（通过预先设置的阈值）。
    #     2. 如果所有足部都没有接触地面，则奖励1，否则奖励为0（或取平均）。
    #     """
    #     feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
    #     # print(feet_indices)
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     # contact_forces: shape = (num_envs, num_bodies, 3)
    #     contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
    #     # 如果所有足部均未接触地面，reward = 1；也可以使用 mean 得到部分奖励
    #     reward = (~contact).float().prod(dim=1)
    #     # print(reward)
    #     return reward
    #     # return 0


    def _reward_handstand_feet_on_air(self):
        """
        改进版：同时检查脚部和膝盖的接触状态
        """
        # 1. 获取脚部索引（原有逻辑）
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 2. 获取膝盖/腿部其他可能接触地面的部位索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]  # 匹配膝盖、大腿、小腿等
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 3. 检查脚部接触
        feet_contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
        
        # 4. 检查膝盖接触
        knee_contact = torch.norm(self.contact_forces[:, knee_indices_tensor, :], dim=-1) > 1.0
        
        # 5. 奖励条件：所有脚部未接触 AND 所有膝盖未接触
        reward = ((~feet_contact).float().prod(dim=1) * 
                (~knee_contact).float().prod(dim=1))
        
        # 6. 根据站立/手倒立命令调整奖励
        standing_command = (self.stand_handstand_state == 0)
        handstand_command = (self.stand_handstand_state == 1)
        
        # 站立命令时，脚部应在地面
        standing_reward = standing_command.float() * (~feet_contact).float().prod(dim=1)
        
        # 手倒立命令时，脚部应在空中
        handstand_reward = handstand_command.float() * (feet_contact).float().prod(dim=1)  # 当所有脚部都接触时奖励
        
        # 最终奖励：根据命令选择适当的奖励
        final_reward = standing_reward + handstand_reward
        
        return final_reward
    
    def _reward_handstand_feet_air_time(self):
        """
        改进版：计算手倒立时足部空中时间奖励，同时惩罚膝盖接触
        """
        threshold = self.cfg.params.handstand_feet_air_time["threshold"]

        # 获取脚部索引
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)
        
        # 获取膝盖索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.device)

        # 计算脚部接触状态
        feet_contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)
        
        # 计算膝盖接触状态
        knee_contact = self.contact_forces[:, knee_indices_tensor, 2] > 1.0  # (batch_size, num_knees)
        any_knee_contact = knee_contact.any(dim=1)  # 任意膝盖接触就惩罚

        # 初始化状态变量（保持原有逻辑）
        if not hasattr(self,"last_contacts") or self.last_contacts.shape != feet_contact.shape:
            self.last_contacts = torch.zeros_like(feet_contact, dtype=torch.bool, device=feet_contact.device)
            
        if not hasattr(self,"feet_air_time") or self.feet_air_time.shape != feet_contact.shape:
            self.feet_air_time = torch.zeros_like(feet_contact, dtype=torch.float, device=feet_contact.device)
        
        # 原有悬空时间计算逻辑
        contact_filt = torch.logical_or(feet_contact, self.last_contacts)
        self.last_contacts = feet_contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        
        # 计算基础悬空时间奖励
        rew_airTime = torch.sum((self.feet_air_time - threshold) * first_contact, dim=1)
        
        # 添加膝盖接触惩罚：有膝盖接触时奖励为0
        rew_airTime = rew_airTime * (~any_knee_contact).float()
        
        # 根据站立/手倒立命令调整奖励
        standing_command = (self.stand_handstand_state == 0)
        handstand_command = (self.stand_handstand_state == 1)
        
        # 站立命令时，脚部应在地面（时间奖励应为0或负值）
        standing_penalty = standing_command.float() * torch.zeros_like(rew_airTime)
        
        # 手倒立命令时，脚部应在空中（时间奖励正常计算）
        handstand_reward = handstand_command.float() * rew_airTime
        
        # 最终奖励：根据命令选择适当的奖励
        final_reward = standing_penalty + handstand_reward
        
        self.feet_air_time *= ~contact_filt
        
        return final_reward

    def _reward_handstand_orientation_l2(self):
        """
        姿态奖励：
        1. 使用 self.projected_gravity（机器人基座坐标系下的重力投影）来评估姿态。
        2. 目标重力方向通过配置传入（例如 [-1, 0, 0] 表示目标为竖直向上）。
        3. 对比当前和目标重力方向的 L2 距离，偏差越大惩罚越大。
        """
        target_gravity = torch.tensor(
            self.cfg.params.handstand_orientation_l2["target_gravity"],
            device=self.device
        )

        # 根据站立/手倒立命令调整目标姿态
        standing_target = torch.tensor([0., 0., -1.], device=self.device)  # 站立时重力向下
        handstand_target = torch.tensor([-1., 0., 0.], device=self.device)  # 手倒立时重力向侧
        
        # 根据命令选择目标姿态
        standing_command = (self.stand_handstand_state == 0).unsqueeze(1).float()
        handstand_command = (self.stand_handstand_state == 1).unsqueeze(1).float()
        
        target_gravity = standing_command * standing_target + handstand_command * handstand_target
        
        return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)
    
    def _reward_joint_smoothness(self):
        """奖励关节运动的平滑性，惩罚剧烈的动作变化"""
        # 1. 动作变化率惩罚（相邻时间步动作差异）
        action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        
        # 2. 关节加速度惩罚
        joint_acceleration = (self.dof_vel - self.last_dof_vel) / self.dt
        joint_accel_penalty = torch.sum(torch.square(joint_acceleration), dim=1)
        
        # 3. 关节加加速度（jerk）惩罚 - 更高级的平滑性
        if hasattr(self, 'last_dof_acc'):
            joint_jerk = (joint_acceleration - self.last_dof_acc) / self.dt
            joint_jerk_penalty = torch.sum(torch.square(joint_jerk), dim=1)
        else:
            joint_jerk_penalty = torch.zeros_like(action_rate_penalty)
        
        # 保存当前加速度供下一帧使用
        self.last_dof_acc = joint_acceleration.clone()
        
        # 组合惩罚项（使用负奖励，因为我们要最小化这些值）
        smoothness_penalty = (
            self.cfg.rewards.joint_smoothness_weights.action_rate * action_rate_penalty +
            self.cfg.rewards.joint_smoothness_weights.acceleration * joint_accel_penalty +
            self.cfg.rewards.joint_smoothness_weights.jerk * joint_jerk_penalty
        )
        
        return -smoothness_penalty  # 返回负值，因为惩罚项越小越好

    def _reward_torque_smoothness(self):
        """奖励扭矩变化的平滑性"""
        if hasattr(self, 'last_torques'):
            torque_change = torch.sum(torch.square(self.torques - self.last_torques), dim=1)
        else:
            torque_change = torch.zeros(self.num_envs, device=self.device)
        
        # 保存当前扭矩供下一帧使用
        self.last_torques = self.torques.clone()
        
        return -torque_change
    
    def _reward_progressive_orientation(self):
        """改进的渐进姿态奖励"""
        current_gravity = torch.nn.functional.normalize(self.projected_gravity, dim=1)
        target_gravity = torch.nn.functional.normalize(self.target_gravity_vec, dim=1)
        
        cos_similarity = torch.sum(current_gravity * target_gravity, dim=1)
        angle_error = torch.acos(torch.clamp(cos_similarity, -0.9999, 0.9999))

        progress = self.transition_progress
        # 确保 tolerance 是 tensor
        tolerance_deg = 30.0 * (1.0 - progress) + 10.0
        tolerance = torch.deg2rad(tolerance_deg)  # progress 是 tensor，所以 tolerance_deg 也是 tensor → OK!

        reward = torch.exp(-(angle_error / tolerance)**2)
        
        standing_command = (self.stand_handstand_state == 0)
        handstand_command = (self.stand_handstand_state == 1)
        
        # 🔧 修复：15.0 必须转为 tensor
        tol_standing = torch.deg2rad(torch.tensor(15.0, device=self.device))
        standing_reward = standing_command.float() * torch.exp(-(angle_error / tol_standing)**2)
        
        handstand_reward = handstand_command.float() * reward
        
        final_reward = standing_reward + handstand_reward
        return final_reward

    def _reward_smooth_transition(self):
        """更强的平滑性奖励"""
        # 关节速度惩罚
        vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)
        
        # 关节加速度惩罚
        acc = (self.dof_vel - self.last_dof_vel) / self.dt
        acc_penalty = torch.sum(torch.square(acc), dim=1)
        
        # 关节加加速度惩罚（jerk）
        jerk = (acc - self.last_dof_acc) / self.dt if hasattr(self, 'last_dof_acc') else torch.zeros_like(acc)
        jerk_penalty = torch.sum(torch.square(jerk), dim=1)
        
        # 保存当前加速度
        self.last_dof_acc = acc.clone()
        
        # 组合惩罚项，加强对剧烈运动的惩罚
        smoothness_penalty = (
            vel_penalty * 0.1 + 
            acc_penalty * 0.05 + 
            jerk_penalty * 0.02
        )
        
        # 根据转换状态调整平滑性奖励
        transition_in_progress = self.transition_in_progress
        
        # 在转换过程中，适当降低平滑性惩罚以允许必要的运动
        adjusted_penalty = transition_in_progress.float() * smoothness_penalty * 0.5 + \
                          (~transition_in_progress).float() * smoothness_penalty
        
        return -adjusted_penalty
    
    def _reward_standing_posture_and_contact(self):
        """改进版：仅处理站立状态下的联合奖励
        - 四足必须全部接触地面
        - 关节角度接近 default_joint_angles
        - 前脚不能过度后摆（相对躯干 X ≥ 阈值）
        - 躯干高度接近 0.32 米
        """
        standing_cmd = (self.stand_handstand_state == 0)
        if not standing_cmd.any():
            return torch.zeros(self.num_envs, device=self.device)

        # === 1. 四足接触地面 ===
        feet_indices_tensor = torch.tensor(self.feet_indices, device=self.device)
        feet_contact = self.contact_forces[:, feet_indices_tensor, 2] > 0.1
        all_feet_on_ground = feet_contact.all(dim=1)

        # === 2. 关节角度接近默认站立姿态 ===
        default_angles_list = [
            -0.02, -0.77, 1.54,  # FL_HIPY, FL_HIPX, FL_KNEE
            0.02, -0.77, 1.54,  # HL_HIPY, HL_HIPX, HL_KNEE
            -0.02, -0.77, 1.54,  # FR_HIPY, FR_HIPX, FR_KNEE
            0.02, -0.77, 1.54,  # HR_HIPY, HR_HIPX, HR_KNEE
        ]
        target_angles = torch.tensor(default_angles_list, device=self.device, dtype=torch.float32)
        angle_error = torch.sum((self.dof_pos - target_angles) ** 2, dim=1)
        posture_reward = torch.exp(-5.0 * angle_error)  # 惩罚尺度：~0.45 rad RMS error → reward≈0.1

        # === 3. 前脚相对位置约束（防止后拖）===
        front_feet_indices = [4, 8]  # FL_FOOT=4, FR_FOOT=8
        base_x = self.root_states[:, 0]  # (N,)
        front_foot_x = self.rigid_body_pos[:, front_feet_indices, 0]  # (N, 2)
        front_foot_rel_x = front_foot_x - base_x.unsqueeze(1)  # (N, 2)

        min_rel_x_threshold = -0.10  # 允许前脚在躯干后方最多 10cm
        front_feet_not_too_back = (front_foot_rel_x >= min_rel_x_threshold).all(dim=1)
        position_reward = front_feet_not_too_back.float()

        # === 4. 躯干高度奖励 ===
        base_z = self.root_states[:, 2]
        target_base_z = 0.32
        base_height_error = torch.abs(base_z - target_base_z)
        base_height_reward = torch.exp(-base_height_error / 0.02)  # 2cm 内高奖励

        # === 5. 组合站立奖励 ===
        standing_reward = (
            all_feet_on_ground.float() *
            posture_reward *
            position_reward *
            base_height_reward
        )

        # === 6. 仅在站立命令下生效 ===
        return standing_cmd.float() * standing_reward
        

    # ==================== 新增奖励函数 ====================

    def _reward_handstand_base_pitch(self):
        """奖励躯干接近竖直（手倒立姿态）"""
        handstand_cmd = (self.stand_handstand_state == 1)
        if not handstand_cmd.any():
            return torch.zeros(self.num_envs, device=self.device)

        # 从 root_states 提取四元数 [x, y, z, w] -> [w, x, y, z]
        quat = self.root_states[:, 3:7]  # [x, y, z, w] in Isaac Gym
        # 转为 [w, x, y, z]
        w, x, y, z = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
        
        # 计算 pitch = arcsin(2*(w*y - z*x))
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))  # [-π/2, π/2]

        # 手倒立理想 pitch ≈ ±π/2
        pitch_error = torch.abs(torch.abs(pitch) - math.pi / 2)
        reward = torch.exp(-pitch_error / 0.3)  # 容忍 ~17° 误差
        return handstand_cmd.float() * reward


    def _reward_handstand_legs_relative_x(self):
        """惩罚前腿相对躯干过度后移（防甩飞）"""
        handstand_cmd = (self.stand_handstand_state == 1)
        mask = handstand_cmd

        # 默认奖励为1（站立时不惩罚）
        reward = torch.ones(self.num_envs, device=self.device)

        if not mask.any():
            return reward

        base_x = self.root_states[:, 0].unsqueeze(1)  # (N, 1)
        front_foot_x = self.rigid_body_pos[:, [4, 8], 0]  # FL_FOOT=4, FR_FOOT=8
        rel_x = front_foot_x - base_x  # 相对于躯干的X位置

        # 允许前脚在躯干前方或略后方（>= -0.3m）
        min_rel_x = -0.3
        excess_backward = torch.relu(min_rel_x - rel_x)  # >0 表示太靠后
        penalty = torch.sum(excess_backward, dim=1)
        reward_val = torch.exp(-penalty / 0.2)

        reward[mask] = reward_val[mask]
        return reward


    def _reward_handstand_hip_symmetry(self):
        """鼓励左右髋关节角度对称"""
        handstand_cmd = (self.stand_handstand_state == 1)
        mask = handstand_cmd
        reward = torch.ones(self.num_envs, device=self.device)

        if not mask.any():
            return reward

        # 假设 DOF 顺序: [FL_HIPY=1, FR_HIPY=7, ...]
        hip_angles = self.dof_pos[:, [1, 7]]
        asymmetry = torch.abs(hip_angles[:, 0] - hip_angles[:, 1])
        sym_reward = torch.exp(-asymmetry / 0.2)
        reward[mask] = sym_reward[mask]
        return reward


    def _reward_handstand_target_height(self):
        """阶段2：精准高度控制"""
        handstand_cmd = (self.stand_handstand_state == 1)
        if not handstand_cmd.any():
            return torch.zeros(self.num_envs, device=self.device)

        stage = self.get_current_stage()
        if stage < 2:
            return torch.zeros_like(handstand_cmd.float())

        foot_z = self.rigid_body_pos[:, [4, 8], 2]  # 前脚高度
        target = self.cfg.params.handstand_feet_height_exp["target_height"]  # 0.75
        error = torch.mean((foot_z - target) ** 2, dim=1)
        reward = torch.exp(-error / 0.1)
        return handstand_cmd.float() * reward


    # ==================== 保留但简化原函数 ====================

    def _reward_handstand_feet_height_exp(self):
        """
        简化版：仅提供基础抬腿奖励（阶段0/1使用）
        不再包含复杂逻辑，避免冲突
        """
        handstand_cmd = (self.stand_handstand_state == 1)
        if not handstand_cmd.any():
            return torch.zeros(self.num_envs, device=self.device)

        # 前脚高度
        front_foot_z = self.rigid_body_pos[:, [4, 8], 2]
        LIFT_THRESHOLD = 0.025
        any_lifted = (front_foot_z > LIFT_THRESHOLD).any(dim=1)

        # 后脚着地
        hind_foot_z = self.rigid_body_pos[:, [12, 16], 2]
        hind_on_ground = (hind_foot_z < 0.05).all(dim=1)

        reward = (any_lifted & hind_on_ground).float()
        return handstand_cmd.float() * reward


    # ==================== 修改课程成功判定 ====================

    def compute_handstand_success_rate(self):
        """
        成功条件：
        - 阶段0: 前脚抬起 + 后脚着地 + 躯干 pitch > 1.2 rad (~70°)
        - 阶段1+: 额外要求后膝离地
        """
        handstand_envs = (self.stand_handstand_state == 1)
        if not handstand_envs.any():
            return 0.0

        stage = self.get_current_stage()

        # 前脚抬起 (>3cm)
        front_foot_height = self.rigid_body_pos[:, [4, 8], 2]
        front_lifted = (front_foot_height > 0.03).all(dim=1)

        # 后脚着地 (<5cm)
        hind_foot_height = self.rigid_body_pos[:, [12, 16], 2]
        hind_on_ground = (hind_foot_height < 0.05).all(dim=1)

        # 躯干姿态：pitch > 1.2 rad
        quat = self.root_states[:, 3:7]
        w, x, y, z = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        upright = torch.abs(pitch) > 1.2  # ~70 degrees

        if stage == 0:
            success = front_lifted & hind_on_ground & upright
        else:
            # 后膝离地 (>5cm)
            hind_knee_height = self.rigid_body_pos[:, [11, 15], 2]
            hind_knee_clear = (hind_knee_height > 0.05).all(dim=1)
            success = front_lifted & hind_on_ground & upright & hind_knee_clear

        final_success = success & handstand_envs
        return final_success.float().mean().item()


    # ==================== 课程阶段更新（带打印） ====================

    def update_curriculum_stage(self):
        """安全更新手倒立课程阶段，并定期打印训练进度"""
        if not hasattr(self, '_handstand_curriculum'):
            self._handstand_curriculum = {
                'stage': 0,
                'thresholds': [0.4, 0.6],  # 阶段0→1: 40%, 阶段1→2: 60%
            }
            print("[课程学习] 初始化手倒立课程：阶段 0")

        current_stage = self._handstand_curriculum['stage']
        if current_stage >= 2:
            if not hasattr(self, '_last_print_step'):
                self._last_print_step = 0
            if self.common_step_counter - self._last_print_step >= 100:
                sr = self.compute_handstand_success_rate()
                print(f"[课程进度] 🟢 阶段 {current_stage}（已完成）| 成功率: {sr:.2%}")
                self._last_print_step = self.common_step_counter
            return

        success_rate = self.compute_handstand_success_rate()
        thresholds = self._handstand_curriculum['thresholds']

        if not hasattr(self, '_last_print_step'):
            self._last_print_step = 0
        if self.common_step_counter - self._last_print_step >= 100:
            print(f"[课程进度] 📊 阶段 {current_stage} | 成功率: {success_rate:.2%} "
                f"(目标: >{thresholds[current_stage]:.0%})")
            self._last_print_step = self.common_step_counter

        if success_rate > thresholds[current_stage]:
            self._handstand_curriculum['stage'] += 1
            new_stage = self._handstand_curriculum['stage']
            desc = ["躯干竖直+前腿抬起", "后膝离地", "目标高度"]
            print(f"[课程学习] 🎯 进入阶段 {new_stage}: {desc[new_stage]} "
                f"(成功率: {success_rate:.2%} > {thresholds[current_stage]:.0%})")


    def get_current_stage(self):
        if not hasattr(self, '_handstand_curriculum'):
            self._handstand_curriculum = {'stage': 0, 'thresholds': [0.4, 0.6]}
        return self._handstand_curriculum['stage']
    
    def _debug_print_env0(self):
        """打印环境 0 的详细状态（用于调试手倒立训练）"""
        if self.common_step_counter % 50 != 0:  # 每50步打印一次，避免刷屏
            return

        env_id = 0
        cmd_state = self.stand_handstand_state[env_id].item()
        stage = self.get_current_stage()

        if cmd_state != 1:
            print(f"[Env0] 🟦 站立模式 | 阶段={stage}")
            return

        # 获取躯干位置和姿态
        base_x = self.root_states[env_id, 0].item()
        quat = self.root_states[env_id, 3:7]
        w, x, y, z = quat[3], quat[0], quat[1], quat[2]
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        # 前脚信息
        left_foot_z = self.rigid_body_pos[env_id, 4, 2].item()   # FL_FOOT
        right_foot_z = self.rigid_body_pos[env_id, 8, 2].item()  # FR_FOOT
        left_foot_x = self.rigid_body_pos[env_id, 4, 0].item()
        right_foot_x = self.rigid_body_pos[env_id, 8, 0].item()

        rel_left_x = left_foot_x - base_x
        rel_right_x = right_foot_x - base_x

        # 后脚/膝盖高度
        hind_foot_z = self.rigid_body_pos[env_id, [12, 16], 2].mean().item()
        hind_knee_z = self.rigid_body_pos[env_id, [11, 15], 2].mean().item()

        # 成功率相关状态
        front_lifted = (left_foot_z > 0.03) and (right_foot_z > 0.03)
        hind_on_ground = hind_foot_z < 0.05
        upright = abs(pitch) > 1.2
        hind_knee_clear = hind_knee_z > 0.05

        success_cond = {
            'front_lifted': front_lifted,
            'hind_on_ground': hind_on_ground,
            'upright': upright,
        }
        if stage >= 1:
            success_cond['hind_knee_clear'] = hind_knee_clear

        success_now = all(success_cond.values())

        # 打印
        print(f"\n{'='*60}")
        print(f"[Env0 Debug] Step={self.common_step_counter} | 阶段={stage} | 命令=手倒立")
        print(f"  躯干: X={base_x:.3f}, Pitch={pitch:.2f} rad ({pitch*180/math.pi:.1f}°)")
        print(f"  前脚高度: 左={left_foot_z:.3f}m, 右={right_foot_z:.3f}m")
        print(f"  前脚相对X: 左={rel_left_x:.3f}m, 右={rel_right_x:.3f}m (阈值 ≥ -0.30)")
        print(f"  后脚高度: {hind_foot_z:.3f}m | 后膝高度: {hind_knee_z:.3f}m")
        print(f"  成功条件: {success_cond} → 当前成功: {success_now}")
        
        # 警告提示
        if rel_left_x < -0.3 or rel_right_x < -0.3:
            print("  💡 警告: 前脚过于靠后！应向前伸手")
        if not upright:
            print("  💡 警告: 躯干未竖直！需调整髋关节角度")
        if not front_lifted:
            print("  💡 建议: 继续抬高前腿（>3cm）")
        
        print(f"{'='*60}\n")


    def _reward_standing(self):
        """站立状态：四足着地 + 身体水平"""
        # 判断是否处于 standing 模式（命令第4维 ≈ 0）
        is_standing_cmd = self.commands[:, 3] < 0.5  # shape: [num_envs]

        # 1. 身体水平
        gravity_error = torch.sum((self.projected_gravity - torch.tensor([0., 0., -1.], device=self.device))**2, dim=1)
        orientation_reward = torch.exp(-gravity_error / 0.02)

        # 2. 四足着地
        feet_z = self.rigid_body_pos[:, self.feet_indices, 2]
        contact_reward = torch.exp(-torch.mean(feet_z, dim=1) / 0.1)

        # 3. 关节接近默认姿态
        joint_error = torch.mean((self.dof_pos - self.default_dof_pos)**2, dim=1)
        posture_reward = torch.exp(-joint_error / 0.01)

        total_reward = (
            orientation_reward * 0.35 +
            contact_reward * 0.3 +
            posture_reward * 0.35
        )

        # ✅ 只在 standing 模式下生效！
        return total_reward * is_standing_cmd.float()


    def _reward_handstand(self):
        is_handstand_cmd = self.commands[:, 3] > 0.5

        # === 原有 reward 不变 ===
        grav_x = self.projected_gravity[:, 0]
        orientation_reward = torch.clamp(-grav_x, 0, 1)

        front_feet_idx = [4, 8]
        front_feet_z = self.rigid_body_pos[:, front_feet_idx, 2]
        lift_reward = torch.mean(front_feet_z, dim=1) / 0.5

        base_x = self.root_states[:, 0:1]
        front_feet_x = self.rigid_body_pos[:, front_feet_idx, 0]
        min_rel_x = torch.min(front_feet_x - base_x, dim=1).values
        leg_back_penalty = torch.relu(-0.05 - min_rel_x)
        position_reward = torch.exp(-leg_back_penalty * 20.0)

        hind_feet_idx = [12, 16]
        contact_forces = self.contact_forces[:, hind_feet_idx, 2]
        support_reward = torch.clamp(torch.mean(contact_forces, dim=1) / 50.0, 0, 1)

        # ✅ 新增：后膝离地奖励（关键！）
        hind_knee_idx = [11, 15]  # 确保这是后膝/小腿
        hind_knee_z = self.rigid_body_pos[:, hind_knee_idx, 2]  # shape: [N, 2]
        min_knee_height = torch.min(hind_knee_z, dim=1).values  # 最低的那只膝盖

        # 要求膝盖至少离地 0.1m（根据你的机器人尺寸调整）
        KNEE_HEIGHT_TARGET = 0.10
        knee_lift_reward = torch.clamp(min_knee_height / KNEE_HEIGHT_TARGET, 0, 1)

        # 或者用指数奖励（更平滑）：
        # knee_error = torch.relu(KNEE_HEIGHT_TARGET - min_knee_height)
        # knee_lift_reward = torch.exp(-knee_error / 0.05)

        total_reward = (
            orientation_reward * 0.2 +
            lift_reward * 0.2 +
            position_reward * 0.2 +
            support_reward * 0.2 +
            knee_lift_reward * 0.2  # ← 新增项！
        )

        return total_reward * is_handstand_cmd.float()
    

    def _reward_feet_air_time(self):
        is_handstand_cmd = self.commands[:, 3] > 0.5
        feet_indices = [4, 8]  # 前脚

        # 检测当前是否接触
        contact = torch.norm(self.contact_forces[:, feet_indices, :], dim=-1) > 1.0
        
        # 【关键】记录接触发生前的空中时间（用于奖励）
        # 如果现在接触了，且上一帧没接触 → 刚结束一次腾空
        contact_transition = contact & (self.last_feet_contact == False)
        
        # 奖励：只在刚落地时发放一次
        air_time_reward = torch.sum(
            torch.clamp(self.feet_air_time - 0.1, min=0.0) * contact_transition,
            dim=1
        )
        
        # 更新空中时间计时器
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact  # 接触时清零
        
        # 更新 last contact 状态
        self.last_feet_contact = contact.clone()
        
        return air_time_reward * is_handstand_cmd.float()
    
    def _reward_low_torques(self):
        torque_sq_sum = torch.sum(torch.square(self.torques), dim=1)
        # 更敏感的衰减（TAU_TARGET 应接近正常值）
        TAU_TARGET = 200.0  # 如果单关节 max=30，12关节满载≈12*900=10800 → 但倒立时应<1000
        return torch.exp(-torque_sq_sum / TAU_TARGET)