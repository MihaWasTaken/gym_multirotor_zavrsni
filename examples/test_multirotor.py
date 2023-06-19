import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from LoggerV2 import LoggerV2
import sys
from gym_multirotor.envs.mujoco.tiltrotor_plus_hover import TiltrotorPlus8DofHoverEnv
import matplotlib.pyplot as plt
import torch


def run(path):
    
    model=torch.load(path)
    
    
    #eval_env=gym.make("TiltrotorPlus8DofHoverEnv-v0")
    eval_env=TiltrotorPlus8DofHoverEnv()
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    obs_new=eval_env.reset_model()
    np.append(obs_new,[0,0,0,0])
    #print(np.shape(obs_new))
    start = time.time()
    logger=LoggerV2(logging_freq_hz=9,
                  output_folder="results",
                  )
    timestamps=[]
    observations=np.ndarray(shape=(0,3))
    reference=np.ndarray(shape=(0,3))
    lin_velocities=np.ndarray(shape=(0,3))
    ang_velocities=np.ndarray(shape=(0,3))
    angles=np.ndarray(shape=(0,4))
    observations_new=[]
    for i in range(50000):
        #path="./results/rl_model_"+str(i*1000)+"_steps"
        #model=PPO.load(path)
        action, _states = model.predict(obs_new,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs_new, reward, done, info = eval_env.step(action)
        #obs_new=eval_env._get_obs
        #print(np.shape(obs_new))
        eval_env.render()
        timestamps.append(i)
        #observations_new.append(eval_env.mujoco_qpos[0:3])
        pos=np.ndarray(shape=0)
        pos=np.append(pos,eval_env.mujoco_qpos[0])
        pos=np.append(pos,eval_env.mujoco_qpos[1])
        pos=np.append(pos,eval_env.mujoco_qpos[2])
        pos=pos.reshape(1,3)
        #pos=np.concatenate((pos,eval_env.mujoco_qpos[0:3]),axis=1)
        observations=np.concatenate((observations,pos),axis=0)
        ref_pos=np.ndarray(shape=0)
        ref_pos=np.append(ref_pos,eval_env.desired_position[0])
        ref_pos=np.append(ref_pos,eval_env.desired_position[1])
        ref_pos=np.append(ref_pos,eval_env.desired_position[2])
        ref_pos=ref_pos.reshape(1,3)
        reference=np.concatenate((reference,ref_pos),axis=0)
        
        ang_vel=np.ndarray(shape=0)
        ang_vel=np.append(ang_vel,obs_new[15:18])
        ang_vel=ang_vel.reshape(1,3)
        ang_velocities=np.concatenate((ang_velocities,ang_vel),axis=0)

        lin_vel=np.ndarray(shape=0) 
        lin_vel=np.append(lin_vel,obs_new[12:15])
        lin_vel=lin_vel.reshape(1,3)
        lin_velocities=np.concatenate((lin_velocities,ang_vel),axis=0)

        ang=np.ndarray(shape=0)
        ang=np.append(ang,obs_new[18:22])
        ang=ang.reshape(1,4)
        angles=np.concatenate((angles,ang),axis=0)
        #res=np.concatenate((reference,eval_env.desired_position[0:3]),axis=0)
        #print(eval_env.desired_position[0])
        #print(obs_new[1])
        #logger.log (timestamp=i,
                    #state= obs,
                    #control=np.zeros(16))
    #eval_env.close()
    #print(reference.item(555))
    print(observations)
    
    obs2=eval_env._get_obs()
    print(obs2)
    print(observations.ndim)
    figure,axis=plt.subplots(3,3)
    axis[0,0].plot(timestamps,observations[:,0])
    axis[0,0].plot(timestamps,reference[:,0])
    axis[0,0].set_title("x_pozicija")
    axis[0,1].plot(timestamps,observations[:,1])
    axis[0,1].plot(timestamps,reference[:,1])
    axis[0,1].set_title("y_pozicija")
    axis[0,2].plot(timestamps,observations[:,2])
    axis[0,2].plot(timestamps,reference[:,2])
    axis[0,2].set_title("z_pozicija")
    axis[1,0].plot(timestamps,ang_velocities[:,0])
    axis[1,0].set_title("x_kutna_brzina")
    axis[1,1].plot(timestamps,ang_velocities[:,1])
    axis[1,1].set_title("y_kutna_brzina")
    axis[1,2].plot(timestamps,ang_velocities[:,2])
    axis[1,2].set_title("z_kutna_brzina")
    axis[2,0].plot(timestamps,lin_velocities[:,0])
    axis[2,0].set_title("x_linearna_brzina")
    axis[2,1].plot(timestamps,lin_velocities[:,1])
    axis[2,1].set_title("y_linearna_brzina")
    axis[2,2].plot(timestamps,lin_velocities[:,2])
    axis[2,2].set_title("z_linearna_brzina")
    figure1,axis1=plt.subplots(2,2)
    axis1[0,0].plot(timestamps,angles[:,0])
    axis1[0,0].set_title("tilt_prvi_motor")
    axis1[1,0].plot(timestamps,angles[:,1])
    axis1[1,0].set_title("tilt_drugi_motor")
    axis1[0,1].plot(timestamps,angles[:,2])
    axis1[0,1].set_title("tilt_treci_motor")
    axis1[1,1].plot(timestamps,angles[:,3])
    axis1[1,1].set_title("tilt_cetvrti_motor")
    plt.show()
    #print(observations.item(555))
    #plt.plot(timestamps,observations,label="mjerena")
    #timestamps1=timestamps
    #plt.plot(timestamps1,reference,label="referentna")
    #plt.xlabel("time step")
    #plt.ylabel("pozicija_x")
    #plt.legend()
    #plt.show()

if __name__=="__main__":
    path=sys.argv[1]
    run(path)
        