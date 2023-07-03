# Import environment libraries
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils import *
# Import preprocessing wrappers
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor, SubprocVecEnv
from matplotlib import pyplot as plt

from utils import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO
from stable_baselines3 import DQN

from matplotlib import pyplot as plt
import torch

SAVE_FREQ = 10000
CHECK_FREQ = 1000
TOTAL_TIMESTEPS = 5000000
BASE_DIR = "C:/Users/erkan/Desktop/EE/e2022_2/EE449/2023/HW3/Code/"

device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device_cuda} is available")
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


def path_creator(model_name, itr):
    CHECKPOINT_DIR = BASE_DIR + f'{model_name}_{itr}/' + 'train/'
    LOG_DIR = BASE_DIR + f'{model_name}_{itr}/' + 'logs/'
    # Create the necessary directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    return CHECKPOINT_DIR, LOG_DIR

def env_generator(CHECKPOINT_DIR, itr):
    # Start the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # Generates the environment
    env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits the joypads moves with important moves
    #startGameRand(env)

    # Apply the preprocessing
    env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
    env = DummyVecEnv([lambda: env])
    #env = SubprocVecEnv([lambda: env])
    # Alternatively, you may use SubprocVecEnv for multiple CPU processors
    if itr == 3 or itr == 6:
        env = VecFrameStack(env, 5, channels_order='last') # Stack frames
    else:
        env = VecFrameStack(env, 4, channels_order='last') # Stack frames
    env = VecMonitor(env, f"{CHECKPOINT_DIR}TestMonitor") # Monitor your progress
    return env

def trainer(model_name, itr):
    CHECKPOINT_DIR, LOG_DIR = path_creator(model_name, itr)
    env = env_generator(CHECKPOINT_DIR, itr)
    callback_func = SaveOnBestTrainingRewardCallback(save_freq=SAVE_FREQ, check_freq=CHECK_FREQ, chk_dir=CHECKPOINT_DIR)

    if itr == 1:
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=256, device=device_cuda)
    elif itr == 2:
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512, device=device_cuda)
    elif itr == 3:
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0000001, n_steps=256, device=device_cuda)

    elif itr == 4:
        model = DQN('CnnPolicy', env, batch_size=192, verbose=1, learning_starts=10000, learning_rate=5e-3, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1, train_freq=8, buffer_size=10000, tensorboard_log=LOG_DIR, device=device_cuda)
    elif itr == 5:
        model = DQN('CnnPolicy', env, batch_size=384, verbose=1, learning_starts=10000, learning_rate=5e-3, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1, train_freq=8, buffer_size=10000, tensorboard_log=LOG_DIR, device=device_cuda)
    elif itr == 6:
        model = DQN('CnnPolicy', env, batch_size=192, verbose=1, learning_starts=10000, learning_rate=5e-4, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1, train_freq=8, buffer_size=10000, tensorboard_log=LOG_DIR, device=device_cuda)
    elif itr == 7:
        model = DQN('CnnPolicy', env, batch_size=192, verbose=1, learning_starts=10000, learning_rate=5e-3, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1, train_freq=8, buffer_size=10000, tensorboard_log=LOG_DIR, device=device_cuda)
    
    else:
        return "Invalid Argument"
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1, callback=callback_func)
    model.save(f'{CHECKPOINT_DIR}best_model_{model_name}_{itr}')


if __name__ == "__main__":
    #trainer("PPO", 1)
    #trainer("PPO", 2)
    #trainer("PPO", 3)
    #trainer("DQN", 4)
    #trainer("DQN", 5)
    #trainer("DQN", 6)
    #trainer("DQN", 7)
    pass