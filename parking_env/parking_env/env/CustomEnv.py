import os
import random
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

steering_angle = 0

class CustomEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, base_path=os.getcwd(), manual=False):
        self.base_path = base_path
        self.manual = manual
        self.car = None
        self.done = False
        self.goal = None
        self.desired_goal = None
        self.target_orientation = None
        self.start_orientation = None
        self.ground = None
        self.road = None
        # 定义状态空间
        obs_low = np.array([-10, -10, -1, -1, -1, -1])
        obs_high = np.array([10, 10, 1, 1, 1, 1])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4种动作：前进、后退、左转、右转
        # 设置小车状态-reward权重
        self.reward_weights = np.array([0.9, 0.9, 0, 0, 0.4, 0.4])
        self.action_steps = 5 # 连续执行五步相同动作
        self.step_threshold = 2000 # 到达2000步后强制跳出当前episode

        # 渲染env
        if render:
            self.client = p.connect(p.GUI) # 可视化物理引擎
            time.sleep(1. / 240.)
        else:
            self.client = p.connect(p.DIRECT)
            time.sleep(1. / 240.)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

    def render(self): # 渲染画面
        p.stepSimulation(self.client)
        time.sleep(1. / 240.)

    def reset(self): # 重置环境
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance = 2,cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0, 0, 0])
        # 加载地面
        self.ground = p.loadURDF(os.path.join(self.base_path, "assets/arena_new.urdf"), basePosition=[0, 0, 0.005], useFixedBase=10)
        self.road1 = p.loadURDF(os.path.join(self.base_path, "assets/road1/urdf/road1.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        self.road2 = p.loadURDF(os.path.join(self.base_path, "assets/road2/urdf/road2.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        self.road3 = p.loadURDF(os.path.join(self.base_path, "assets/road3/urdf/road3.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        self.road4 = p.loadURDF(os.path.join(self.base_path, "assets/road4/urdf/road4.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        self.road5 = p.loadURDF(os.path.join(self.base_path, "assets/road5/urdf/road5.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        self.road6 = p.loadURDF(os.path.join(self.base_path, "assets/road6/urdf/road6.urdf"), basePosition=[0, 0, 0.01], useFixedBase=10)
        # 停车位
        # p.addUserDebugLine([-0.05, -0.22, 0.02], [0.265, -0.22, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([0.265, -0.22, 0.02], [0.475, -0.55273, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([0.475, -0.55773, 0.02], [0.115, -0.55773, 0.02], [0.98, 0.98, 0.98], 2.5)
        # p.addUserDebugLine([0.115, -0.55773, 0.02], [-0.05, -0.22, 0.02], [0.98, 0.98, 0.98], 2.5)
        # target = [0.18, -0.28954]
        # target_orientation = np.pi*2/3

        basePosition = [-0.3, 0, 0.1] # 小车初始位置
        self.goal = np.array([0.19632, -0.38339]) # 小车目标点
        self.target_orientation = np.pi * 2 / 3 # 小车目标旋转角
        self.start_orientation = [0, 0, np.pi] # 小车初始旋转角
        # 将小车的目标observation拼接
        self.desired_goal = np.array([self.goal[0], self.goal[1], 0.0, 0.0, np.cos(self.target_orientation), np.sin(self.target_orientation)])

        # 加载小车
        self.t = Car(self.client, basePosition=basePosition, baseOrientationEuler=self.start_orientation, action_steps=self.action_steps)
        self.car = self.t.car

        # 获取当前observation
        car_ob, self.vector = self.t.get_observation()
        observation = np.array(list(car_ob))
        self.step_cnt = 0

        return observation

    # 小车与目标点的距离
    def distance_function(self, pos): 
        return np.sqrt(pow(pos[0] - self.goal[0], 2) + pow(pos[1] - self.goal[1], 2))
    
    # 计算当前步的reward
    def compute_reward(self, achieved_goal, desired_goal, info): 
        p_norm = 0.5
        reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.reward_weights)), p_norm)

        return reward

    # 判断是否碰撞
    def judge_collision(self):
        done = 0
        points1 = p.getContactPoints(self.car, self.road1)
        points2 = p.getContactPoints(self.car, self.road2)
        points3 = p.getContactPoints(self.car, self.road3)
        points4 = p.getContactPoints(self.car, self.road4)
        points5 = p.getContactPoints(self.car, self.road5)
        points6 = p.getContactPoints(self.car, self.road6)
        if len(points1):
            done = 1
        if len(points2):
            done = 2
        if len(points3) or len(points5):
            done = 3
        if len(points4):
            done = 4
        if len(points6):
            done = 5
        return done

    # 小车执行一步action
    def step(self, action):
        self.t.apply_action(action)  # 小车执行动作
        p.stepSimulation()
        car_ob, self.vector = self.t.get_observation()  # 获取小车状态

        position = np.array(car_ob[:2])
        distance = self.distance_function(position)
        reward = self.compute_reward(car_ob, self.desired_goal, None)

        if self.manual:
            print(f'dis: {distance}, reward: {reward}, center: {self.goal}, pos: {car_ob}')

        self.done = False
        self.success = False

        if distance < 0.05:
            reward = 1000
            self.success = True
            self.done = True

        self.step_cnt += 1
        if self.step_cnt > self.step_threshold:  # 限制episode长度为step_threshold
            self.done = True
        if self.judge_collision() == 1:  # 碰撞
            reward = -2000
            self.done = True
        if self.judge_collision() == 2:  # 碰撞
            reward = -1000
            self.done = True
        if self.judge_collision() == 3:  # 碰撞
            reward = -500
            self.done = True
        if self.judge_collision() == 4:  # 碰撞
            reward = 100
            self.done = True
        if self.judge_collision() == 5:  # 碰撞
            reward = -200
            self.done = True
        if self.done:
            self.step_cnt = 0

        observation = np.array(list(car_ob))

        info = {'is_success': self.success}

        return observation, reward, self.done, info

    # 设置环境种子
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 关闭当前环境
    def close(self):
        p.disconnect(self.client)

# 初始化小车
class Car:
    def __init__(self, client, base_path=os.getcwd(), basePosition=[0, 0, 0.1], baseOrientationEuler=[0, 0, np.pi],
                 max_velocity=2, max_force=50, action_steps=None):
        """
        初始化小车

        :param client: pybullet client
        :param basePosition: 小车初始位置
        :param baseOrientationEuler: 小车初始方向
        :param max_velocity: 最大速度
        :param max_force: 最大力
        :param action_steps: 动作步数
        """

        self.client = client
        self.base_path = base_path
        # 加载小车模型
        self.car = p.loadURDF(os.path.join(self.base_path, "assets/test.SLDASM/urdf/test.SLDASM.urdf"), basePosition=basePosition, baseOrientation=p.getQuaternionFromEuler(baseOrientationEuler))

        self.max_velocity = max_velocity
        self.max_force = max_force
        self.action_steps = action_steps

    # 小车执行动作函数
    def apply_action(self, action):
        global steering_angle # 当前前轮转向角
        velocity = self.max_velocity
        force = self.max_force
        streer = [3, 7] # 转向关节
        wheel = [4, 8] # 前轮从动
        motor = [0, 1] # 后轮驱动
        for i in wheel:
            p.setJointMotorControl2(self.car, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        if action == 0:  # 前进
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=velocity, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=-velocity, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.stepSimulation()
        elif action == 1:  # 后退
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=-velocity, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=velocity, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.stepSimulation()
        elif action == 2:  # 左转
            if steering_angle > -np.pi / 4:  # 最大转向角
                steering_angle -= np.pi / 20
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 3, p.POSITION_CONTROL, targetPosition=steering_angle) # steering_angle是转向角
                p.setJointMotorControl2(self.car, 7, p.POSITION_CONTROL, targetPosition=-steering_angle)
                p.stepSimulation()
        elif action == 3:  # 右转
            if steering_angle < np.pi / 4: # 最大转向角
                steering_angle += np.pi / 20            
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=force) # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 3, p.POSITION_CONTROL, targetPosition=steering_angle) # steering_angle是转向角
                p.setJointMotorControl2(self.car, 7, p.POSITION_CONTROL, targetPosition=-steering_angle)
                p.stepSimulation()
        elif action == 4:  # 停止
            targetVel = 0
            for i in range(0,8):
                p.setJointMotorControl2(self.car, i, p.VELOCITY_CONTROL, targetVelocity=targetVel,force=force)
                                        
            p.stepSimulation()
        # elif action == 5:  # 回正方向盘
        #     for i in range(self.action_steps):
        #         p.setJointMotorControl2(self.car, 3, p.POSITION_CONTROL, targetPosition=0)
        #         p.setJointMotorControl2(self.car, 7, p.POSITION_CONTROL, targetPosition=0)
        #         p.stepSimulation()
        #     p.stepSimulation()
        else:
            raise ValueError

    # 获取小车当前observation
    def get_observation(self):
        position, angle = p.getBasePositionAndOrientation(self.car)  # 获取小车位姿
        angle = p.getEulerFromQuaternion(angle)
        velocity = p.getBaseVelocity(self.car)[0]

        position = [position[0], position[1]]
        velocity = [velocity[0], velocity[1]]
        orientation = [np.cos(angle[2]), np.sin(angle[2])]
        vector = angle[2]

        observation = np.array(position + velocity + orientation)  # 拼接坐标、速度、角度

        return observation, vector
