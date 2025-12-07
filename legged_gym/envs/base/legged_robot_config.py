from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 46 # 扩展命令维度后，观测维度从45增加到46
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 #leg [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False # True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False # 是否启用课程学习，如果为 True，会从简单的命令开始（如小速度），逐渐增加难度
        max_curriculum = 1. # 表示最大难度系数，用于缩放命令范围
        num_commands = 4 # 扩展命令维度：从4增加到5 (lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_handstand)
        resampling_time = 10. # time before command are changed[s] 每10秒重新生成一次命令
        heading_command = False # if true: compute ang vel command from heading error 使用朝向误差计算角速度
        class ranges:
            lin_vel_x = [-0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            
            stand_handstand = [0, 1]  # 站立/手倒立命令：0表示站立，1表示手倒立

    class init_state:
        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': -0.02,   # [rad]
            'HL_HipX_joint': 0.02,   # [rad]
            'FR_HipX_joint': -0.02,  # [rad]
            'HR_HipX_joint': 0.02,   # [rad]

            'FL_HipY_joint': -0.77,     # [rad]
            'HL_HipY_joint': -0.77,   # [rad]
            'FR_HipY_joint': -0.77,     # [rad]
            'HR_HipY_joint': -0.77,   # [rad]

            'FL_Knee_joint': 1.54,   # [rad]
            'HL_Knee_joint': 1.54,    # [rad]
            'FR_Knee_joint': 1.54,  # [rad]
            'HR_Knee_joint': 1.54,    # [rad]
        }

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "foot" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["SHANK", "THIGH","Knee"]
        terminate_after_contacts_on = ["TORSO",]
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            # ===== 惩罚项（务必小！）=====
            termination = -25.0          # 正确
            collision = -2.0             # OK，可保留
            torques = 2e-7              # ✅ 合理（若 torque²_sum ~1e5，则单步≈-2） e-6
            dof_acc = 0              # ✅ 很小，安全
            dof_vel = -0.0005
            dof_pos_limits = -0.4
            action_rate = -0.0007       # ✅ 合理（动作差值~0.1，平方和~0.01 → -2e-5）
            torque_smoothness = 0   # ⚠️ 可能偏大！建议先降

            # ===== 主任务奖励 =====
            standing = 21              # OK
            handstand = 26.25             # OK
            handstand_feet_air_time = 0.0  # ✅ 注意：这个函数必须返回“空中时间”（正数）
            low_torques = 0.0  # 正权重！鼓励小扭矩

            # ===== 其他（设为0很安全）=====
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.0
            base_height = -0.0
            feet_stumble = -0.0
            stand_still = 0.0
            


        class joint_smoothness_weights:
            action_rate = 1.0      # 动作变化率权重
            acceleration = 0.5     # 加速度权重  
            jerk = 0.2            # 加加速度权重
            
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma) 0.25
        soft_dof_pos_limit = 0.8 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.9
        base_height_target = 0.95
        max_contact_force = 100. # forces above this value are penalized

    class params:  # 参数单独放在params类中
        handstand_feet_height_exp = {
            "target_height": 0.75,
            "std": 0.4
        }
        handstand_orientation_l2 = {
            "target_gravity": [-1, 0.0, 0.0]
        }
        handstand_feet_air_time = {
            "threshold": 5.0
        }
        feet_name_reward={
            "feet_name" : "F.*_FOOT"
        }
        
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 2 # 1.5
            lin_vel = 0.2 # 0.1
            ang_vel = 0.4 #0.2
            gravity = 0.05 #0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 100000 # number of policy updates
        normalize_reward: True

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt 



