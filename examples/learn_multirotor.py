import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import envs
from stable_baselines3.common.env_checker import check_env
from gym_multirotor.envs.mujoco.tiltrotor_plus_hover import TiltrotorPlus8DofHoverEnv
from stable_baselines3.a2c import MlpPolicy
import os
from datetime import datetime
import numpy as np
import time
from Logger import Logger
from stable_baselines3.common.callbacks import CheckpointCallback

def run():
    env=gym.make("TiltrotorPlus8DofHoverEnv-v0")
    
    check_env(env,
              warn=True,
              skip_render_check=True
              )
    
    model=PPO(MlpPolicy,
              env,
              verbose=1)
    
    checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./results/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    #model.learn(total_timesteps=10000,callback=checkpoint_callback)
    model.learn(total_timesteps=10000)
    model.save("results")



if __name__ == "__main__":
    run()
    

