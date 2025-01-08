import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
from jsac.helpers.logger import Logger
# from jsac.envs.rl_chemist.env import RLChemistEnv
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from utils import combine_images, image_render

import time
# import multiprocessing as mp
from non_visual_reacher import FrankaPandaEnv, run_manual
import argparse
import shutil
import numpy as np
import cv2

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--name', default='rl_chemist', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default='prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='rl_chemist', type=str)
    parser.add_argument('--image_height', default=84, type=int)
    parser.add_argument('--image_width', default=84, type=int)
    parser.add_argument('--image_history', default=3, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1_000_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=10000, type=int)
    parser.add_argument('--env_steps', default=2_000_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=True, action='store_true')
    # parser.add_argument('--apply_rad', default=True, action='store_true')
    # parser.add_argument('--rad_offset', default=0.01, type=float)
    parser.add_argument('--global_norm', default=1.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=3e-5, type=float)
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.01, type=float)

    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)
    
    # encoder
    parser.add_argument('--spatial_softmax', default=True, action='store_true')
    
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    
    # misc
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=500_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    # render
    parser.add_argument('--render_interactive', default=False, type=bool)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--record', default=False, type=bool)

    args = parser.parse_args()
    return args


def run_training(args):
    env = FrankaPandaEnv(render=args.render_interactive)
    env = WrappedEnv(env, start_step= args.start_step, 
                     start_episode=args.start_episode, episode_max_steps=1000)
    
    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
            print('Previous work dir removed.')
        elif inp == '':
            pass
        else:
            exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)

    L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    if args.save_model:
        make_dir(args.model_dir)

    proprioception_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env_action_space = env.action_space

    print(f"{proprioception_shape}, {action_shape}, {env_action_space}")
    set_seed_everywhere(seed=args.seed)
    
    # args.single_image_shape = (args.image_width, args.image_height, 3)
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.image_shape = None
    args.net_params = config
    agent = SACRADAgent(vars(args))
    obs = env.reset()
    first_step = True
    task_start_time = time.time()

    while env.total_steps < args.env_steps:
        t1 = time.time()
        action = agent.sample_actions(obs)
        t2 = time.time()
        next_obs, reward, done, info = env.step(action)
        # truncate = env.total_steps % max_episode_length == 0
        # done = done or truncate
        t3 = time.time()
        mask = 1.0 if not done or 'truncated' in info else 0.0
        agent.add(obs, action, reward, next_obs, mask, first_step)
        obs = next_obs
        first_step = False
        
        if done or 'truncated' in info:
            
            obs = env.reset()
            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - task_start_time
            info['dump'] = True
            L.push(info)
            first_step = True
        if env.total_steps > args.init_steps and env.total_steps % args.update_every == 0:

            update_infos = agent.update()
            
            if update_infos is not None:


                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False

                    L.push(update_info)
                    # exit(0)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)
    if args.save_model:
        agent.checkpoint(env.total_steps)
    L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')

def run_policy(args):
    env = FrankaPandaEnv(render=args.render_interactive)
    env = WrappedEnv(env, start_step= args.start_step, 
                     start_episode=args.start_episode, episode_max_steps=1000)
    
    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'

    if os.path.exists(os.path.abspath(args.work_dir)):
        print("loading model from work directory")
    else:
        exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)

    L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    if args.save_model:
        make_dir(args.model_dir)

    proprioception_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env_action_space = env.action_space

    print(f"{proprioception_shape}, {action_shape}, {env_action_space}")
    set_seed_everywhere(seed=args.seed)
    
    # args.single_image_shape = (args.image_width, args.image_height, 3)
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.image_shape = None
    args.net_params = config
    agent = SACRADAgent(vars(args))
    obs = env.reset()
    returns = []
    # cap = cv2.VideoCapture(0)
    # _, frame = cap.read()
    image = image_render(env)

    if image is not None:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
        # _, frame = cap.read()

    # final_image = combine_images(image_bgr, frame)
    height, width, _ = image_bgr.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter("results/sim/run.mp4", fourcc, fps=30, frameSize=(width, height))

    for _ in range(1):
        done = False
        rewards = []
        while not done:
            action = agent.sample_actions(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            image = image_render(env)

            if image is not None:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
                video_writer.write(image_bgr)
                cv2.imshow('Env Visuals', image_bgr)
            # _, frame = cap.read()

            # final_image = combine_images(image_bgr, frame)
            # video_writer.write(final_image)
            
            # cv2.imshow('Final Image', final_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rewards.append(reward)
            obs = next_obs
            if done:
                obs = env.reset()
                time.sleep(2)
                print("Episode Done")
        returns.append(sum(rewards))

    np.savetxt("results/sim_episodic_returns.txt", returns)
    cv2.destroyAllWindows()
    video_writer.release()
    env.close()
    print("Policy Run Complete")
    exit(0)     
    

if __name__=="__main__":
    args = parse_args()
    if args.load_model < 0:
        run_training(args)
    else:
        run_policy(args)
    
    