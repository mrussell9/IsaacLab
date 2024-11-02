"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--estimator", action="store_true", default=False, help="Learn estimator during training.")
parser.add_argument("--symmetric", action="store_true", default=False, help="Enforce value symmetry during training.")
parser.add_argument("--wandb", action="store_true", default=False, help="Plot with model from WandB.")
parser.add_argument("--wandb_run", type=str, default="", help="Run from WandB.")
parser.add_argument("--wandb_model", type=str, default="", help="Model from WandB.")
# append RSL-RL cli arguments
cli_args.add_rma_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.num_envs = 1
args_cli.video = True
args_cli.device = "cuda:0"  # "cpu"  # cpu does work for contacts for some reason
args_cli.wandb = True if "" not in [args_cli.wandb_run, args_cli.wandb_model] else args_cli.wandb
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import shutil
import torch
from datetime import datetime

from isaaclab.rma.frameworks.robot_rl.runners import BasePolicyRunner, AdaptionModuleRunner

Runner = BasePolicyRunner

from isaaclab.rma.frameworks.robot_rl.utils import Plotter

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
import isaaclab.rma.tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rma_cfg(args_cli.task, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # specify directory for logging experiments
    ext_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_root_path = os.path.join(ext_path, "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    if agent_cfg.resume:
        resume_path = args_cli.checkpoint # get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
    elif args_cli.wandb:
        # load the policy
        resume_path = ""
        env_cfg = None
        # load configuration
        run_path = args_cli.wandb_run
        if run_path == "":
            run_path = input(
                "\033[96mEnter the Weights and Biases run path located on the Overview panel; i.e"
                " usr/Spot-Blind/abc123\033[0m\n"
            )
        while True:
            model_name = args_cli.wandb_model
            if model_name == "":
                model_name = input(
                    "\n\033[96mEnter the name of the model file to download; i.e model_100.pt \n"
                    + "Press Enter again without a file name to quit.\033[0m\n"
                )
            if model_name == "":
                return
            if model_name[:6] != "model_":
                model_name = "model_" + model_name
            if model_name[-3:] != ".pt":
                model_name += ".pt"
            try:
                resume_path, env_cfg = cli_args.pull_policy_from_wandb(log_root_path, run_path, model_name)
                print(f"\033[92m\n[INFO] added policy to load\033[0m")
                break
            except Exception:
                print(
                    "\n\033[91m[WARN] Unable to download from Weights and Biases, is the path"
                    " and filename correct?\033[0m"
                )
        model_file_dir = os.path.dirname(resume_path)
        model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        plot_dir = os.path.join(log_dir, "plots")
        model_dir = os.path.join(log_dir, run_path.split("/")[-1])
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = Runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.resume or args_cli.wandb:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    latent_policy = ppo_runner.get_latents_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    timestep_start = 0
    # H = 1  # TODO add history logic back in
    # create a plotter
    log = Plotter(env.unwrapped, store_dynamics=False)
    env_idx = env.unwrapped.cfg.viewer.env_index
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            latents = latent_policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)
            obs_flat = obs[env_idx].flatten().tolist()
            if timestep >= timestep_start:
                log.log(
                    lin_vel=obs_flat[0:3],
                    ang_vel=obs_flat[3:6],
                    proj_g=obs_flat[6:9],
                    vel_cmd=obs_flat[9:12],
                    q=obs_flat[12:24],
                    dq=obs_flat[24:36],
                    action=actions[0].flatten().tolist(),
                    torque=env.unwrapped.scene["robot"].data.applied_torque[0].flatten().tolist(),
                    # q_des=env.unwrapped.scene["robot"].data.joint_pos_target[0].flatten().tolist(),
                    q_des=None,  # calculate q_des from Ka*a explicitly in the plotter
                    latents=latents[env_idx].flatten().tolist()
                )
        timestep += 1
        # Exit the play loop after recording logs
        if timestep == args_cli.video_length:
            break
    # save logs
    log.plot(plot_dir)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()