"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--phase", type=int, required=True, help="Phase of RMA to train")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=12_000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb", action="store_true", default=False, help="Plot with model from WandB.")
parser.add_argument("--wandb_run", type=str, default="", help="Run from WandB.")
parser.add_argument("--wandb_model", type=str, default="", help="Model from WandB.")
# append RSL-RL cli arguments
cli_args.add_rma_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.video = True
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
import torch
import shutil
from datetime import datetime

from isaaclab.rma.frameworks.robot_rl.runners import BasePolicyRunner, AdaptionModuleRunner

assert args_cli.phase in [1,2], "Phase argument must be set to 1 or 2"

if args_cli.phase == 1:
    Runner = BasePolicyRunner
elif args_cli.phase == 2:
    model_test = False
    if args_cli.resume or args_cli.wandb:
        model_test = True
    assert model_test == True, "Running phase 2 requires a base policy. Please specify a model to load by passing \
                                --resume or --wandb"
    Runner = AdaptionModuleRunner

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
import isaaclab.rma.tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rma_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RMA agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rma_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rma", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = Runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = args_cli.checkpoint #resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

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
                model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
                model_dir = os.path.join(log_dir, run_path.split("/")[-1])
                os.makedirs(model_dir, exist_ok=True)
                shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
                break
            except Exception:
                print(
                    "\n\033[91m[WARN] Unable to download from Weights and Biases, is the path"
                    " and filename correct?\033[0m"
                )

    if agent_cfg.resume or args_cli.wandb:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
