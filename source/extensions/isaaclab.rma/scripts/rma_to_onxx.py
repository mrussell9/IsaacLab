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
cli_args.add_rma_args(parser)
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

import os
import torch
from torch import nn

from datetime import datetime

import cli_args

from isaaclab.rma.frameworks.robot_rl.modules import RMA1, RMA2
from isaaclab.rma.frameworks.robot_rl.utils import export_RMA_policy_as_onnx

def main(base_model, adaption_module):
    run_path = ""

    log_root_path = os.path.join("logs", "rma", "merged_models")
    log_root_path = os.path.abspath(log_root_path)

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"[INFO]: LOADING BASE POLICY FROM WANDB")
    base_model_path, base_policy_env_cfg = cli_args.load_wandb_policy(run_path = "Spot_RMA/runs/jdnmetb6", model_name = "49999", 
                                                                    log_root_path = log_root_path + "/base_policies", log_dir = log_dir)

    print(f"[INFO]: LOADING ADAPTION MODULE FROM WANDB")
    adaption_module_path, adaption_module_env_cfg = cli_args.load_wandb_policy(run_path = "Spot_RMA_Phase2/runs/n6qcnflh", model_name = "79998", 
                                                                            log_root_path = log_root_path + "/adaption_modules", log_dir = log_dir)

    base_model_dict = torch.load(base_model_path)
    adaption_module_dict = torch.load(adaption_module_path)

    base_model.load_state_dict(base_model_dict['model_state_dict'])
    adaption_module.load_state_dict(adaption_module_dict['model_state_dict'])

    export_RMA_policy_as_onnx(base_model, adaption_module, log_root_path)

if __name__ == '__main__':
    base_model = RMA1(num_actor_obs=249,
                    num_critic_obs=249,
                    num_actions=12,
                    env_size=201,
                    prev_step_size=48,
                    z_size=30,
                    actor_hidden_dims=[512, 256, 128],
                    encoder_hidden_dims=[256, 128, 30],
                    critic_hidden_dims=[512, 256, 128],
                    activation="elu",
                    init_noise_std=1.0,
                    )

    adaption_module = RMA2(num_actions=12,
                        env_size=48,
                        prev_step_size=48,
                        z_size=30,
                        encoder_hidden_dims=[512, 256, 128, 32],
                        conv_params=[[32, 32, 8, 4], [32, 32, 5, 1], [32, 32, 5, 1]],
                        activation="elu",
                        history_length=50,
                        )
                        
    main(base_model, adaption_module)

