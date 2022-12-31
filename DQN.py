import argparse
import datetime
import os

import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

env_name = 'parking_env-v0'
render = True
total_step = 10000000
save_frequency = 50000
log_path = './log'

time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M')
log_path = os.path.join(log_path, f'DQN_{time}')
ckpt_path = os.path.join(log_path, f'dqn_agent')

env = gym.make(env_name, render=render)
model = DQN('MlpPolicy', env, verbose=1, seed=0)
# model = DQN.load(args.ckpt_path, env)
env.reset()
logger = configure(log_path, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=save_frequency, save_path=log_path, name_prefix='dqn_agent')
model.learn(total_timesteps=total_step, callback=checkpoint_callback)
model.save(ckpt_path)
del model
