#imports
from tkinter.tix import Tree
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation 
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from trainer import TrainAndLoggingCallback


#create game env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#simplfy controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#greyscale 
env = GrayScaleObservation(env, keep_dim=True)
#dummy env
env = DummyVecEnv([lambda: env])
#stack frames
env = VecFrameStack(env, 4, channels_order='last')


state = env.reset()
state, reward, done, info = env.step([5])

CheckpointDir = './mario/saved_models'
LogsDir = './mario/logs'

#callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CheckpointDir)

#create the model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LogsDir, learning_rate=0.0001, n_steps=512)

#train the model
model.learn(total_timesteps=20000,callback=callback)

#save model
model.save('./mario/saved_models/testModel')

#load model
model = PPO.load('./mario/saved_models/best_model_20000')

# Start the game 
state = env.reset()
# Loop through the game
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

#NOTE: process has to be killed via task manager 