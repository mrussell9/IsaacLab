"""Script to convert trained policies for deployment."""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert a checkpoint of an RL agent from RSL-RL to onnx for deployment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--estimator", action="store_true", default=False, help="Load estimator during conversion.")
parser.add_argument("--symmetric", action="store_true", default=False, help="Enforce value symmetry during training.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 1
args_cli.video = False
args_cli.device = "cpu"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import traceback
from datetime import datetime

import carb
import gymnasium as gym
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)
# from rsl_rl.runners import OnPolicyRunner
# from rsl_rl.runners import TactileOnPolicyRunner, VisionOnPolicyRunner
from isaaclab.rma.frameworks.robot_rl.runners import OnPolicyRunner

Runner = OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
import isaaclab.rma.tasks  # noqa: F401
import isaaclab.rma.mdp as rma_mdp

import torch

def convert_policy():
    """Convert pytorch policies to onnx at different checkpoints."""

    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments
    export_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    export_root_path = os.path.abspath(export_root_path)
    print(f"[INFO] Logging experiment in directory: {export_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    export_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        export_dir += f"_{agent_cfg.run_name}"
    export_dir = os.path.join(export_root_path, "exported", export_dir)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load the policy
    resume_path = ""
    env_cfg = None

    # load configuration
    policy_list = []
    while True:
        policy_load_type = input(
            "\n\033[96mSelect a policy loading option to add to the conversion queue:\033[0m \n"
            + "\033[95m(1) to convert using an explicit path to policy .pt file\033[0m \n"
            + "\033[95m(2) to convert policy from Weights and Biases\033[0m \n"
        )
        # catch invalid input string
        if len(policy_load_type) > 1:
            continue
        match policy_load_type:
            case "1":
                while True:
                    policy_path = input(
                        "\n\033[96mEnter the path of the policy model file you wish to convert one at a time\n"
                        + "Press Enter without a path to finish and convert all added policies.\033[0m\n"
                    )
                    if policy_path == "":
                        break
                    resume_path = os.path.abspath(policy_path)
                    if os.path.exists(resume_path):
                        env_cfg = cli_args.load_local_cfg(resume_path)
                        policy_list.append(tuple([resume_path, env_cfg]))
                        print(f"\033[92m\n[INFO] added policy to conversion queue of length {len(policy_list)}\033[0m")
                    else:
                        print(
                            "\n\033[91m[WARN] Got invalid file path, unable to add selected file to conversion"
                            " queue!\033[0m"
                        )
                break
            case "2":
                run_path = input(
                    "\033[96mEnter the weights and biases run path located on the Overview panel; i.e"
                    " usr/Spot-Blind/abc123\033[0m\n"
                )
                while True:
                    model_name = input(
                        "\n\033[96mEnter the name of the model file to download one at a time; i.e model_100.pt \n"
                        + "Press Enter again without a file name to finish and convert all policies in queue.\033[0m\n"
                    )
                    if model_name == "":
                        break
                    try:
                        resume_path, env_cfg = cli_args.pull_policy_from_wandb(export_dir, run_path, model_name)
                        policy_list.append(tuple([resume_path, env_cfg]))
                        print(f"\033[92m\n[INFO] added policy to conversion queue of length {len(policy_list)}\033[0m")
                    except Exception:
                        print(
                            "\n\033[91m[WARN] Unable to download from Weights and Biases for conversion, is the path"
                            " and filename correct?\033[0m"
                        )
                break

    for idx in range(len(policy_list)):
        resume_path, env_cfg = policy_list[idx]
        model_file_dir = os.path.dirname(resume_path)
        model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # create runner from rsl-rl
        #ppo_runner = VisionOnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        #ppo_runner = TactileOnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        ppo_runner = Runner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        #obs = torch.zeros(1, 51, device=agent_cfg.device)
        #image_obs = torch.zeros(1, 2, 53, 30, device=agent_cfg.device)
        #print("actions: ",ppo_runner.alg.actor_critic(obs, image_obs))
        #print("actions: ",ppo_runner.alg.actor_critic.act_inference(obs))
        ppo_runner.load(resume_path, load_optimizer=False)

        export_model_dir = os.path.join(model_file_dir, f"{model_file_name}_deployment")
        os.makedirs(export_model_dir)
        print(f"[INFO]: Saving env config json file to {export_model_dir}")
        cfg_save_path = os.path.join(export_model_dir, "env_cfg.json")
        # with open(cfg_save_path, "w") as fp:
        #     json.dump(env_cfg, fp, indent=4)
        print(f"[INFO]: Saving policy onnx file to {export_model_dir}")
        export_estimator_policy_as_onnx(ppo_runner.alg.actor_critic, ppo_runner.estimator_obs_normalizer, export_model_dir, filename=f"{model_file_name}.onnx")
    if len(policy_list) > 0:
        print(f"\n\033[92m[INFO] Exported {len(policy_list)} policy(ies) to {export_dir}\033[0m")


if __name__ == "__main__":
    # run the main execution
    convert_policy()
    # close sim app
    simulation_app.close()
