# Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3 #
## 关于本文 ##
### 介绍 ### 
Legged_gym_handstand_for_DeepRobotics_Lite3 是希望通过强化学习的方式训练一种可以让四足机器人手倒立的方法，本文的最终环节是将训练好的模型部署到云深处绝影 Lite3 机器人上(sim2real)。本文不会注重算法的介绍，而是更倾向于流程的跑通，从模型训练(train)到初步仿真(play)到仿真测试(sim2sim test)再到实机部署(sim2real)。笔者使用 RTX 4060 显卡的笔记本电脑，能够正常跑通上述流程。本文的很多内容都需要一个良好的网络环境才可以正常访问和下载，推荐在网络环境良好的条件下进行实验。
### 参考和引用内容 ###
本文的部分内容基于下面的内容进行修改或使用人工智能搜索生成得到，或直接引用了其中的图片或文本，在此表示感谢。
[1]https://github.com/cmjang/legged_gym_handstand
[2]https://za8k8pe2ezm.feishu.cn/wiki/N5hFwIrC3isrVckQRRPcx6cHnPs
[3]https://github.com/DeepRoboticsLab/gamepad
[4]https://github.com/leggedrobotics/rsl_rl
[5]https://github.com/leggedrobotics/legged_gym
[6]https://github.com/fan-ziqi/rl_sar
[7]https://github.com/fan-ziqi/robot_lab
[8]https://github.com/DeepRoboticsLab/Lite3_rl_deploy
## 初步 ##
### 1. 使用 python 3.6、3.7 或 3.8 创建新的 python 虚拟环境（推荐 3.8） ###
```bash
conda create -n myenv python=3.8
```
### 2. 使用 cuda-11.3 安装 pytorch 1.10： ###
需要注意的是如果你的电脑有安装有报错，或者在之后的运行中提到了有关 cuda 的报错，不妨试试安装新一点的 cuda 版本，比如 cuda 13 的版本，笔者是如此操作的。由于每个人的电脑环境各不相同，笔者不在此过多介绍如何安装环境，如果遇到报错不妨询问 ai 的帮助。
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
### 3. 安装 IsaacGym ###
从 [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) 下载并安装 Isaac Gym Preview 3，事实上现在一部分的强化学习已经使用更新的 IsaacLab 替代，但是不影响 IsaacGym 的正常下载和使用，legged_gym 也是使用此进行训练的。
```bash
cd isaacgym/python && pip install -e .
```
 尝试运行示例 `cd examples && python 1080_balls_of_solitude.py` ，若运行中出现了下面的报错不妨试试：
```bash
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
conda activate env_isaacgym
```
1. 找到conda环境中的libpython库路径
	```bash
	find ~/miniconda3/envs/env_isaacgym -name "libpython3.8.so.1.0" 2>/dev/null
	```
2. 设置LD_LIBRARY_PATH（根据实际路径调整）
	```bash
	export LD_LIBRARY_PATH=~/miniconda3/envs/env_isaacgym/lib:$LD_LIBRARY_PATH
	```
### 4. 安装 rsl_rl (PPO 实现)   ###
克隆 [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
```bash
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```
### 5. 安装 legged_gym ###
克隆 [https://github.com/leggedrobotics/legged_gym.git](https://github.com/leggedrobotics/legged_gym.git)
```bash
cd legged_gym && pip install -e .
```
### 6. 安装 Lite3_rl_deploy ###
克隆 [https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git](https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git)
下面的操作推荐退出 conda 环境进行，否则可能后续启动 mujoco 可能会报错。
退出 conda 环境的指令是：
```bash
conda deactivate
```
退出之后的效果是终端前面没有“括号”。然后我们执行 segmentation debug 工具安装，这个工具可以帮助你快速定位问题出现在哪里（虽然我不知道有什么用）。
```bash
# segmentation debug 工具安装
sudo apt-get install libdw-dev
wget https://raw.githubusercontent.com/bombela/backward-cpp/master/backward.hpp
sudo mv backward.hpp /usr/include
```
然后让我们安装一下 python 依赖，这很有可能相当于在你的非 conda 环境内“再”装一个 python 环境，这往往可能导致空间的占用，但是我尝试过使用 conda 环境之后的操作会报错，所以这里推荐在 非 conda 环境下安装依赖。
```bash
# 依赖安装 (python3.10)
pip install pybullet "numpy < 2.0" mujoco
git clone --recurse-submodule https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git
```
然后让我们 cd 到 Lite3_rl_deploy 目录，创建一个 build 文件夹并 cd 进入 build 尝试编译。如果提示 build 文件夹已经存在不妨删除 build 文件夹然后再重新创建，包括在 sim2real 过程中也有可能需要删除 build 重新编译。删除 build 文件夹的指令是：
```bash
rm -rf ./build
```

```bash
# 编译
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DSEND_REMOTE=OFF
# 指令解释
# -DBUILD_PLATFORM：电脑平台，Ubuntu为x86，机器狗运动主机为arm
# -DBUILD_SIM：是否使用仿真器，如果在实机上部署设为OFF 
make -j
```
## 03. 使用 legged_gym_handstand 训练动作 ##
### 克隆仓库 ###
克隆 [https://github.com/XiaoBaiBZS/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3.git](https://github.com/XiaoBaiBZS/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3.git)
### legged_gym_handstand 修改内容 ###
```tree
.
├── legged_gym
│   ├── envs
│   │   ├── a1
│   │   │   ├── a1_config.py
│   │   │   └── __pycache__
│   │   │       └── a1_config.cpython-38.pyc
│   │   ├── anymal_b
│   │   │   ├── anymal_b_config.py
│   │   │   └── __pycache__
│   │   │       └── anymal_b_config.cpython-38.pyc
│   │   ├── anymal_c
│   │   │   ├── anymal.py
│   │   │   ├── flat
│   │   │   │   ├── anymal_c_flat_config.py
│   │   │   │   └── __pycache__
│   │   │   │       └── anymal_c_flat_config.cpython-38.pyc
│   │   │   ├── mixed_terrains
│   │   │   │   ├── anymal_c_rough_config.py
│   │   │   │   └── __pycache__
│   │   │   │       └── anymal_c_rough_config.cpython-38.pyc
│   │   │   └── __pycache__
│   │   │       └── anymal.cpython-38.pyc
│   │   ├── base
│   │   │   ├── base_config.py
│   │   │   ├── base_task.py
│   │   │   ├── legged_robot_config.py
│   │   │   ├── legged_robot.py
│   │   │   └── __pycache__
│   │   │       ├── base_config.cpython-38.pyc
│   │   │       ├── base_task.cpython-38.pyc
│   │   │       ├── legged_robot_config.cpython-38.pyc
│   │   │       └── legged_robot.cpython-38.pyc
│   │   ├── cassie
│   │   │   ├── cassie_config.py
│   │   │   ├── cassie.py
│   │   │   └── __pycache__
│   │   │       ├── cassie_config.cpython-38.pyc
│   │   │       └── cassie.cpython-38.pyc
│   │   ├── __init__.py
│   │   ├── lite3
│   │   │   ├── lite3_config.py
│   │   │   └── __pycache__
│   │   │       └── lite3_config.cpython-38.pyc
│   │   └── __pycache__
│   │       └── __init__.cpython-38.pyc
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-38.pyc
│   ├── scripts
│   │   ├── play.py
│   │   └── train.py
│   ├── tests
│   │   └── test_env.py
│   └── utils
│       ├── helpers.py
│       ├── __init__.py
│       ├── logger.py
│       ├── math.py
│       ├── __pycache__
│       │   ├── helpers.cpython-38.pyc
│       │   ├── __init__.cpython-38.pyc
│       │   ├── logger.cpython-38.pyc
│       │   ├── math.cpython-38.pyc
│       │   ├── task_registry.cpython-38.pyc
│       │   └── terrain.cpython-38.pyc
│       ├── task_registry.py
│       └── terrain.py
├── legged_gym.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── LICENSE
├── licenses
│   ├── assets
│   │   ├── a1_license.txt
│   │   ├── ANYmal_b_license.txt
│   │   ├── ANYmal_c_license.txt
│   │   └── cassie_license.txt
│   └── dependencies
│       └── matplotlib_license.txt
├── logs
│   ├── onnx
│   │   └── legged.onnx
│   └── rough_lite3
│       ├── exported
│       │   └── policies
│       │       ├── policy_001.pt
│       │       ├── policy_002.pt
│       │       ├── policy_003.pt
│       │       ├── policy_004.pt
│       │       ├── policy_005.pt
│       │       └── policy_006.pt
│       ├── Oct30_12-08-11_
│       │   └── events.out.tfevents.1761797292.bai.22565.0
│       ├── Oct30_12-27-40_
│       │   ├── events.out.tfevents.1761798460.bai.23578.0
│       │   ├── model_0.pt
│       ├── Oct30_13-48-32_
│       │   ├── events.out.tfevents.1761803312.bai.27051.0
│       │   └── model_0.pt
│       └── Oct30_13-48-57_
│           ├── events.out.tfevents.1761803338.bai.27122.0
│           ├── model_0.pt
├── README.md
├── resources
│   ├── actuator_nets
│   │   └── anydrive_v3_lstm.pt
│   └── robots
│       ├── a1
│       │   ├── a1_license.txt
│       │   ├── meshes
│       │   │   ├── calf.dae
│       │   │   ├── hip.dae
│       │   │   ├── thigh.dae
│       │   │   ├── thigh_mirror.dae
│       │   │   ├── trunk_A1.png
│       │   │   └── trunk.dae
│       │   └── urdf
│       │       └── a1.urdf
│       ├── anymal_b
│       │   ├── ANYmal_b_license.txt
│       │   ├── meshes
│       │   │   ├── anymal_base.dae
│       │   │   ├── anymal_foot.dae
│       │   │   ├── anymal_hip_l.dae
│       │   │   ├── anymal_hip_r.dae
│       │   │   ├── anymal_shank_l.dae
│       │   │   ├── anymal_shank_r.dae
│       │   │   ├── anymal_thigh_l.dae
│       │   │   ├── anymal_thigh_r.dae
│       │   │   ├── base_uv_texture.jpg
│       │   │   └── carbon_uv_texture.jpg
│       │   └── urdf
│       │       └── anymal_b.urdf
│       ├── anymal_c
│       │   ├── ANYmal_c_license.txt
│       │   ├── meshes
│       │   │   ├── base.dae
│       │   │   ├── base.jpg
│       │   │   ├── battery.dae
│       │   │   ├── battery.jpg
│       │   │   ├── bottom_shell.dae
│       │   │   ├── bottom_shell.jpg
│       │   │   ├── depth_camera.dae
│       │   │   ├── depth_camera.jpg
│       │   │   ├── drive.dae
│       │   │   ├── drive.jpg
│       │   │   ├── face.dae
│       │   │   ├── face.jpg
│       │   │   ├── foot.dae
│       │   │   ├── foot.jpg
│       │   │   ├── handle.dae
│       │   │   ├── handle.jpg
│       │   │   ├── hatch.dae
│       │   │   ├── hatch.jpg
│       │   │   ├── hip.jpg
│       │   │   ├── hip_l.dae
│       │   │   ├── hip_r.dae
│       │   │   ├── lidar_cage.dae
│       │   │   ├── lidar_cage.jpg
│       │   │   ├── lidar.dae
│       │   │   ├── lidar.jpg
│       │   │   ├── remote.dae
│       │   │   ├── remote.jpg
│       │   │   ├── shank.jpg
│       │   │   ├── shank_l.dae
│       │   │   ├── shank_r.dae
│       │   │   ├── thigh.dae
│       │   │   ├── thigh.jpg
│       │   │   ├── top_shell.dae
│       │   │   ├── top_shell.jpg
│       │   │   ├── wide_angle_camera.dae
│       │   │   └── wide_angle_camera.jpg
│       │   └── urdf
│       │       └── anymal_c.urdf
│       ├── cassie
│       │   ├── cassie_license.txt
│       │   ├── meshes
│       │   │   ├── abduction_mirror.stl
│       │   │   ├── abduction.stl
│       │   │   ├── achilles-rod.stl
│       │   │   ├── hip_mirror.stl
│       │   │   ├── hip.stl
│       │   │   ├── knee-output_mirror.stl
│       │   │   ├── knee-output.stl
│       │   │   ├── pelvis.stl
│       │   │   ├── plantar-rod.stl
│       │   │   ├── shin-bone_mirror.stl
│       │   │   ├── shin-bone.stl
│       │   │   ├── tarsus_mirror.stl
│       │   │   ├── tarsus.stl
│       │   │   ├── thigh_mirror.stl
│       │   │   ├── thigh.stl
│       │   │   ├── toe_mirror.stl
│       │   │   ├── toe-output-crank.stl
│       │   │   ├── toe.stl
│       │   │   ├── torso.stl
│       │   │   ├── yaw_mirror.stl
│       │   │   └── yaw.stl
│       │   └── urdf
│       │       └── cassie.urdf
│       └── lite3
│           ├── meshes
│           │   ├── fl_hip.STL
│           │   ├── fl_shank_collision.STL
│           │   ├── fl_shank.STL
│           │   ├── fl_thigh.STL
│           │   ├── fr_hip.STL
│           │   ├── fr_shank_collision.STL
│           │   ├── fr_shank.STL
│           │   ├── fr_thigh.STL
│           │   ├── hl_hip.STL
│           │   ├── hl_shank_collision.STL
│           │   ├── hl_shank.STL
│           │   ├── hl_thigh.STL
│           │   ├── hr_hip.STL
│           │   ├── hr_shank_collision.STL
│           │   ├── hr_shank.STL
│           │   ├── hr_thigh.STL
│           │   └── torso.STL
│           └── urdf
│               └── lite3.urdf
└── setup.py

```
以上是我为了 Deepobotics Lite3 做适配之后的文件树，这个仓库是基于 cmjiang 的仓库修改而成，主要修改的内容有：
1. 增加了 DeepoRbotics JueYing Lite3 的模型用于训练。
2. 自以为是的修改了一些强化学习的参数，导致训练效果不是很好，需要再继续调整。
3. 修改了训练中的输入维度，删除了地形高度测量维度(170维度)、删除了基座线速度维度(3维度)，现在为45维度输入，12维度输出，使得更加易于 sim2sim test 和 sim2real 。
4. 修改了 play.py 中的部分内容，使得导出时可以生成 onnx 文件，以便于 sim2sim test 和 sim2real 。
下面将会以修改的内容对整个仓库进行介绍，也欢迎大家提出更好的建议来完善功能。
### Lite3 Asset ###
1. 首先可以从云深处的官方 Github repo 中翻到云深处曾经使用 IsaacGym 训练四足机器人的例程，在这个例程中可以翻到 Lite3 的 urdf 文件和 meshes 文件夹。需要注意的是，也许你可以在云深处的官方渠道获取到不同的名字相同的 urdf 文件，但是他们可能存在些许差异，可能会造成模型显示异常。我们需要将 urdf 文件和 meshes 文件夹放置在和 a1 同级的目录下。当你发现运行仿真训练的时候模型出现了异常，不妨试试以下几点：
	1. 要不就试试用我的 repo 中的 urdf 文件呢？
	2. `robot_config`文件下`flip_visual_attachments =True或False`。
	3. 换一个 urdf 文件。
2. 接着我们需要参考着 a1 的配置文件编写 lite3 的配置文件，我们在 envs 目录下创建 lite3 的目录，lite3 的目录应当与 a1 同级。然后我们在 lite3 目录下面创建 lite3_config.py 。这个文件描述了机器人关节的初始位置、关节初始角度、控制参数、资源文件、奖励函数等。这个文件可以参考 a1 机器人的并作修改实现。
	1. 初始位置：pos = [0.0, 0.0, 0.42] # x,y,z [m] 描述了机器人训练时候的初始位置，这里的 z 值我设置的是 0.42 ，但是可以稍微低一点，设置为0.35，看起来似乎会更好一点。
	2. 初始化关节信息：a1 机器人的关节命名和 lite3 的机器人关节命名有很多差异，比如膝关节在 a1 的命名是 FL_calf_joint ，但是在 lite3 中是 FL_Knee_joint ，同时对关节前后的命名也有差异，需要更改。他们的初始值可以参考云深处官方的 rl repo，比如 Lite3_rl_training 获得，或者之前云深处书中提及的使用 IsaacGym 训练的 repo 。当然，你也可以使用 ai 来辅助你得到这些值。
	3. control 里面的信息我没有进行修改，似乎这样就可以，也是官方例程里面描述的。
	4. asset 这里需要对应修改为自己的 lite3 目录下的 urdf 文件。
	5. foot_name 描述了足部的定义，这里他应该是使用正则匹配去匹配 urdf 文件中包含 foot （不区分大小写）的节点。
	6. penalize_contacts_on 描述了哪些位置在与地面接触的时候会受到奖励惩罚。这里需要根据 urdf 文件中的描述修改
	7. terminate_after_contacts_on 描述了哪些位置在与地面接触的时候会终止训练回合，与上一个仅有扣分惩罚不同，这里是直接结束调训练回合。我们设置为机器人的基座。
	8. experiment_name 描述了实验的名称，按需修改即可。
3. 至此，我们“导入” Lite3 的相关内容就算结束了，接着我们会对修改后的 legged_robot_config.py 进行介绍。
### legged_robot_config.py ###
#### env ####
1. num_envs 描述了同步训练环境的数量，如果是想在训练的时候进行可视化初步看一下，这个值不应该设置过大否则电脑会渲染的时候卡住。我在无头模式(--headless)下训练一般将此值设置为 4096 。
2. num_observations 这个值表示输入的观测维度，这个值原先是 235 ，被我修改为了 45 ，如此修改的目的是更方便的进行云深处官方的 repo Lite3-rl-deploy 来实现 sim2sim test (仿真-仿真测试) 和 sim2real (仿真-实机部署)。下面是关于维度的一些介绍：
	1. 基座线速度 3维度 从235维度中删除，因为基座线速度在实机中很容易漂移，偏差较大，所以一般不用
	2. 基座角速度 3维度 保留
	3. 控制命令 3维度 保留
	4. 投影重力 3维度 保留
	5. 关节位置 12维度 保留
	6. 关节速度 12维度 保留
	7. 动作 12维度 保留
	8. 高度扫描 170维度 从235维度中删除，因为此处训练手倒立不需要识别地形
	如此以来，最开始的235维度就变成了剩下的45维度了。关于如何“删除”提到的维度，将在后续 legged_robot.py 中介绍，此处仅仅是把数字进行更改。
3. num_privileged_obs 如果是 None 则 step 返回的 privileged_obs 是 None，如果不是则返回 privileged_obs (一般在使用非对称网络结构时使用)。
4. num_actions 是模型输出维度表述了机器人关节自由度。
5. env_spacing 描述了机器人模型在生成时候的间距。
6. episode_length_s 表示了机器人最多存活的时间。
#### rewards ####
这里描述了奖励函数的奖励值，可以尝试修改这里的参数进行优化效果。需要注意的是我的 repo 中的设置参数不是最佳参数，甚至可能无法正常训练出不错的模型，由于本文仅仅是介绍从训练到部署的流程跑通，所以不过多介绍参数的设置。
1. tracking_lin_vel 表示线速度
2. tracking_ang_vel 表示角速度
3. handstand_feet_height_exp 描述了高权重​​鼓励倒立时脚部达到目标高度
4. ​​handstand_feet_on_air 描述了倒立时脚部离地的奖励
5. handstand_feet_air_tim 描述了倒立时脚部悬空时间的奖励
6. ​​torques = -0.00001​​：惩罚关节扭矩使用（节能）
7. dof_acc = -2.5e-7​​：惩罚关节加速度（平滑运动）
8. collision = -1.​​：惩罚碰撞（安全性）
9. action_rate = -0.01​​：惩罚动作变化率（动作平滑性）
其他不过多介绍。
#### params ####
这里描述了一些动作的“评判指标”。
1. handstand_feet_height_exp 中 target_height 描述了希望立起来的时候足部到达的高度；std 表示的是开始得到奖励的容许误差，即当其足部达到一定高度的时候就逐渐获得奖励，这个值设置的越大，越容易获得到奖励，设置的越小，对奖励获得的标准越严格，可能会限制智能体探索。
2. handstand_orientation_l2 表述了机器人基座坐标系下的重力投影来评估姿态，[1, 0, 0] 表示目标为竖直向上；同理，当你训练其彻底反过来的倒立时候不妨改变“足部的定义” feet_name 为 `"H.*_FOOT"`和这里的值为 [-1, 0, 0]
3. handstand_feet_air_time 表述了当机器人足部保持悬空多少秒才获得奖励得分。
4. feet_name_reward 表述了希望抬起地面的“足部”的定义，这里需要参考 urdf 文件进行设置，同时这里是一个正则表达式，这里的 F 即代表希望希望前腿抬起。当你设置为 H 即代表希望后腿抬起。
####  runner ####
max_iterations 设置了训练的最大轮数，一般设置的越大训练花费的时间越多，需要进行权衡。
### legged_robot.py ###
这个文件表述了对之前配置文件中的一些内容的具体实现，本文不做详细介绍，仅对修改的位置作以解释。
1. `compute_reward` 函数中删除基座线速度 base_lin_vel 的维度，213行附近、删除地形高度扫描相关内容，224行附近。
2. `_get_noise_scale_vec` 函数中对噪声的维度进行了删除，479行附近。
3. `_init_buffers` 函数中作了些许修改以适配之前的修改，515行附近。
### play.py ###
我们运行仿真训练后可以通过运行 play.py 进行仿真查看，但是我们在训练结束后往往只能获得到 .pt 文件，这显然是不易于使用云深处官方提供的 repo Lite3_rl_deploy 进行 sim2sim test 和 sim2real 的，因为这个 repo 需要使用 .onnx 文件。那我们需要将 .pt 文件转化成 .onnx 文件，尽管云深处的 repo 中提供了一个 pt2onnx.py 的脚本来帮我们进行转化，但是实际操作下来，我是没有运行成功的，让 ai 帮我修改代码也是没有成功。

按照云深处工程师的说法“您好，我们的pt2onnx.py只是一个范例文件。那里的输入输出只是一个例子，并没有真实意义。我们这里的policy输入是45维输出是12维。您在rl training训练完play之后会自动导出一个onnx文件，您直接把那个onnx文件复制粘贴过来就好。”

于是，接下来将会对 play.py 等代码进行修改，实现生成 .onnx 文件。
1. 首先在 helpers.py 中新增函数 export_policy_as_onnx，这个函数中的保存 onnx 文件的路径需要根据自己的需求进行修改。  需要注意的是这里的 torch.onnx.export 中的 input_names 和 output_names 不要写错。![[Pasted image 20251030220056.png]]
 ```python
def export_policy_as_onnx(actor_critic):
	model = copy.deepcopy(actor_critic.actor).to('cpu')
	actor_input = torch.randn(1, 45) # 根据实际情况调整形状
	body_onnx_path = '/home/bai/legged_gym_handstand/logs/onnx/' + 'legged.onnx'
	paths = [body_onnx_path]
	for path in paths:
		if os.path.exists(path):
			os.remove(path)
			print(f"已删除: {path}")
		else:
			print(f"文件不存在: {path}")
	print("path:",body_onnx_path)
	torch.onnx.export(model, actor_input, body_onnx_path,input_names=['obs'],output_names=['actions'], opset_version=11)
   ```
2. 然后对 `__init__.py ` 等进行修改，使其 import 。然后别忘记保存，笔者这里更改完后没有保存，导致运行的时候提示 import 失败， debug 了半天。![[Pasted image 20251030220157.png]]![[Pasted image 20251030220219.png]]
3. 在 play.py 中 if EXPORT_POLICY: 处增加 export_policy_as_onnx 的函数调用。![[Pasted image 20251030220348.png]]
4. 如此，便实现了当运行 play.py 的时候在对应目录下生成 .onnx 文件了。

## 04. train、play、sim2sim test and sim2real ##
### train ###
```bash
cd legged_gym_handstand/legged_gym/script
python train.py --task=lite3 --headless
```
首先先 cd 进入代码文件夹，然后运行 train.py 即可，传入参数 task=lite3 ，可选使用无头模式运行，使用无头模式运行能够显著提高训练效率，推荐加上 --headless 。

若运行中出现了下面的报错不妨试试：
```bash
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
conda activate env_isaacgym
```
1. 找到conda环境中的libpython库路径
	```bash
	find ~/miniconda3/envs/env_isaacgym -name "libpython3.8.so.1.0" 2>/dev/null
	```
2. 设置LD_LIBRARY_PATH（根据实际路径调整）
	```bash
	export LD_LIBRARY_PATH=~/miniconda3/envs/env_isaacgym/lib:$LD_LIBRARY_PATH
	```

若运行中出现了找不到 lite3 任务的情况，不妨把前文提到的关于 lite3 的修改同步复制到 legged_gym/resources 和legged_gym/legged_gym/env 里面。笔者遇到了类似的问题，如此修改便解决，但是不清楚是否由于此引起。
### play ###
```bash
cd legged_gym_handstand/legged_gym/script
python play.py --task=lite3
```
首先先 cd 进入代码文件夹，然后运行 play.py 即可，传入参数 task=lite3 ，运行完成后即可看到训练的效果，然后就可以在对应目录下找到生成的 .onnx 文件。
## sim2sim test
如果在上一步骤中获得到的效果比较理想，可以进行 sim2sim test ，将直接训练好的模型部署到实机上往往容易失败，而且比较危险，很有可能弄坏设备，因此我们先进行 sim2sim test ，mujoco 是一个与真实环境比较类似的仿真环境，我们需要使用由云深处提供的 repo ，进行 sim2sim test ，这一步的操作推荐在非 conda 环境内进行，前文已经提及，笔者在 conda 环境中执行 sim2sim test 会报错。我们将得到的 .onnx 文件复制到 Lite3_rl_deploy/policy/ppo 文件夹中，然后修改 .onnx 文件的名称为 policy.onnx ，然后我们打开两个终端，一个终端启动仿真 mujoco ，另一个启用键盘控制。
```bash
cd Lite3_rl_deploy

# 终端1 (mujoco)
cd interface/robot/simulation
python3 mujoco_simulation.py

# 终端2 
cd build
./rl_deploy
```
使用键盘 `z` 键来让 mujoco 中的四足机器人站立，然后使用 `c` 键来切换四足机器人为 rl 模式，使用 `w/a/s/d` 来控制前后左右平移，使用 `q/e` 来顺逆时针旋转。通过 mujoco 即可看到四足机器人的 sim2sim 效果，如果效果比较不错，可以尝试进行下一步实机部署。

若发现当四足机器人执行关键动作的时候由于系统误认为其处在危险动作而进入关节阻尼状态，可以再仿真中关闭此功能，即任何状态下都为力控状态(尽管这比较危险)，具体操作是打开文件Lite3_rl_deploy/state_machine/state_machine.hpp，190 行附近执行y以下操作，注释掉条件判断，直接使用控制器的状态切换。修改完成后 cd 进入 build 目录重新 make -j 编译即可。需要确保在仿真条件下无误后再进行 sim2real 的部署，否则会在实机出现问题(危险状况下)仍然力控，这样很用可能损坏设备甚至伤人。
```cpp
// if(current_controller_->LoseControlJudge()) next_state_name_ = StateName::kJointDamping;
// else next_state_name_ = current_controller_ -> GetNextStateName();

// 直接使用控制器的状态切换
next_state_name_ = current_controller_ -> GetNextStateName();
```
### sim2real ###
首先先让我们将电脑与四足机器人连接到同一个网络，即电脑连接四足机器人的 wifi ，通常是 ysc- 开头的网络。密码通常是 12345678 。然后我们使用 scp 传输数据。
```bash
# scp传输文件 (打开本地电脑终端)
scp -r ~/Lite3_rl_deploy ysc@192.168.2.1:~/
```
然后我们使用 ssh 连接机器狗，密码通常是英文单引号 `'` 。
```bash
# ssh连接机器狗运动主机以远程开发，密码有以下三种组合
#Username	Password
#ysc		' (a single quote)
#user		123456 (推荐)
#firefly	firefly
ssh ysc@192.168.2.1
# 输入密码后会进入远程开发模式
```
然后 cd 到 Lite3_rl_deploy 的文件夹。并按照下面的操作进行编译，推荐先将 build 目录删除，重新编译。否则可能会报错。
```bash
# 编译
cd Lite3_rl_deploy

# 删除 build 文件夹（可选）
rm -rf ./build

# 编译
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF
# 指令解释
# -DBUILD_PLATFORM：电脑平台，Ubuntu为x86，机器狗运动主机为arm
# -DBUILD_SIM：是否使用仿真器，如果在实机上部署设为OFF 
make -j 
./rl_deploy
```
执行完上述指令后四足机器人应当关节回零然后我们拿出手柄，连接机器狗的 wifi ，然后掌机上需要提前安装好云深处为强化学习测试开发的遥控 App “云深处科技”，应用图标是蓝色的 Flutter 图标。![[Pasted image 20251031115100.png]]然后我们进入 App ，点击左上角更多按钮，然后修改端口号为：192.168.1.120:12121 ，然后点击右上角保存退出重新进入应用。![[image/Pasted image 20251031115136.png]]
然后我们回到电脑，通过 ssh 连接四足机器人，cd 进入并使用 vim 打开 jy_exe/conf/network.toml 文件，修改数据上报的 ip ，我们希望其返回关节信息、传感器信息到我们的电脑，修改此处的 ip 为 192.168.2.1 。然后我们试着使用手柄让四足机器人起立，我们可以按下手柄右侧字母 A 键，可以看到四足机器人起立，为平地站立静止状态。然后我们可以在确保安全的条件下按下手柄右侧字母 Y 键，此时机器人将会进入到 rl 力控状态。若不希望继续进行，可以 ctrl+c 终止 ./rl_deploy 进行。

