# 系统基础库
import time
import numpy as np
# 强化学习库
import gym, parking_env
from stable_baselines3 import DQN
import pybullet as p
# 位姿检测库
import cv2
# 小车控制
import requests

def transPosition(x, y, center=[1500, 360]): # 将摄像头的xy坐标转换为模型的xy坐标
    x = (x - center[0]) / 1897.4
    y = (y - center[1]) / 1897.4
    return x, y

def transPointing(x1, y1, x2, y2): # 计算小车朝向
    mol = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) 
    x_cos = y2 - y1 / mol
    x_sin = x2 - x1 / mol
    return x_cos, x_sin

def doAction(action, steering_angle): # 根据模型返回的action操控小车
    if action == 0:  # 前进
        r = requests.post('http://192.168.31.194/motor_control?speed=60')
        r = requests.post('http://192.168.31.194/servo_control?angle='+str(steering_angle))
    elif action == 1:  # 后退
        r = requests.post('http://192.168.31.194/motor_control?speed=-60')
        r = requests.post('http://192.168.31.194/servo_control?angle='+str(steering_angle))
    elif action == 2:  # 左转
        if steering_angle > -45:
            steering_angle -= 9
        r = requests.post('http://192.168.31.194/motor_control?speed=60')
        r = requests.post('http://192.168.31.194/servo_control?angle='+str(steering_angle))
    elif action == 3:  # 右转
        if steering_angle < 45:
            steering_angle += 9
        r = requests.post('http://192.168.31.194/motor_control?speed=60')
        r = requests.post('http://192.168.31.194/servo_control?angle='+str(steering_angle)) 
    elif action == 4:  # 停止
        r = requests.post('http://192.168.31.194/motor_control?speed=0')
        r = requests.post('http://192.168.31.194/servo_control?angle=0'+str(steering_angle))
    return steering_angle
    
if __name__ == 'main':
    '''
    小车位姿检测
    '''
    # 初始化小车位姿检测和小车控制
    red_x = 0
    red_y = 0
    blue_x = 0
    blue_y = 0
    lastPos_x = 0
    lastPos_y = 0
    steering_angle = 0
    threshold = 1000
    cnt = 0
    #初始化摄像机
    cap = cv2.VideoCapture('http://192.168.31.80:4747//mjpegfeed?1920x1080')

    '''
    加载强化学习模型
    '''
    action_list = [] # 储存模型返回的动作
    ckpt_path = 'log/dqn_agent_5900000_steps.zip'
    # Evaluation
    env = gym.make("parking_env-v0", render=False)
    obs = env.reset()
    model = DQN.load(ckpt_path, env=env, print_system_info=True)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    episode_return = 0
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        position, _ = p.getBasePositionAndOrientation(env.car)
        p.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-80,cameraTargetPosition=position)
        time.sleep(1 / 240)

        episode_return += reward
        if done:
            break
    env.close()
    print(f'episode return: {episode_return}')

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        # 转换颜色空间 BGR 到 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # HSV 值范围设定
        red_inf = np.array([150, 40, 50])
        red_sup = np.array([180, 255, 255])
        blue_inf = np.array([100, 43, 46])
        blue_sup = np.array([124, 255, 255])
        # 设置HSV的阈值使得只取某种颜色
        red_mask = cv2.inRange(hsv, red_inf, red_sup)  # 获取遮罩
        blue_mask = cv2.inRange(hsv, blue_inf, blue_sup)

        # 将掩膜和图像逐像素相加
        red = cv2.bitwise_and(frame, frame, mask=red_mask)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        # 查找轮廓
        # CV_RETR_EXTERNAL 只检测最外围轮廓
        # CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
        red_cnts, hrc = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        blue_cnts, hrc = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # 绘制轮廓
        cv2.drawContours(red, red_cnts, -1, (0, 0, 255), 20)
        cv2.drawContours(blue, blue_cnts, -1, (0, 0, 255), 20)
        # 排除干扰，计算位置，绘制方框
        r_x, r_y, r_w, r_h = 0, 0, 0, 0
        b_x, b_y, b_w, b_h = 0, 0, 0, 0
        for red_cnt in red_cnts:
            red_area = cv2.contourArea(red_cnt)
            # print(area)
            if red_area > threshold:  # 过滤阈值
                r_x, r_y, r_w, r_h = cv2.boundingRect(red_cnt)
                lastPos_x = red_x
                lastPos_y = red_y
                red_x = int(r_x + r_w / 2)
                red_y = int(r_y + r_h / 2)
                cv2.rectangle(frame, (r_x, r_y), (r_x + r_w, r_y + r_h), (0, 0, 255), 20)
                #  cv2.circle(frame, (red_x, red_y), 20, (0, 0, 255), 20) # 绘制方框中心
        for blue_cnt in blue_cnts:
            blue_area = cv2.contourArea(blue_cnt)                
            if blue_area > threshold:
                b_x, b_y, b_w, b_h = cv2.boundingRect(blue_cnt)
                lastPos_x = blue_x
                lastPos_y = blue_y
                blue_x = int(b_x + b_w / 2)
                blue_y = int(b_y + b_h / 2)
                cv2.rectangle(frame, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0), 20)
                #  cv2.circle(frame, (blue_x, blue_y), 20, (0, 0, 255), 20) # 绘制方框中心
                # 将小车在图像上的坐标转换为在模型上的坐标
        position = transPosition(red_x, red_y)
        cv2.putText(frame, "({:0<4f}, {:0<4f})".format(position[0], position[1]),
                (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)  # 文字
        cv2.imshow('position', frame)
        # 小车的旋转角
        orientation = transPointing(red_x, red_y, blue_x, blue_y)
        # 将小车的实际observation转换为模型上observation
        obs[0] = position[0]
        obs[1] = position[1]
        obs[4] = orientation[0]
        obs[5] = orientation[1]
        # 绘制小车朝向箭头
        cv2.arrowedLine(frame, (red_x, red_y), (blue_x, blue_y), (0, 255, 0), \
                    thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.1)
        steering_angle = doAction(action[cnt], steering_angle)
        cnt += 1          


