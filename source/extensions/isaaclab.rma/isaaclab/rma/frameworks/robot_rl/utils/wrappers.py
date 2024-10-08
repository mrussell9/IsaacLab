from dataclasses import MISSING

import copy
import torch
import os

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class RmaActorCriticsCfg(RslRlPpoActorCriticCfg):
    encoder_hidden_dims: list[int] = MISSING

def export_policy_as_onnx(actor_critic: object, path: str, filename="policy.onnx", verbose=False) -> None:
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        filename: The name of exported onnx file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""

class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(self, x) -> torch.Tensor:
        return self.actor(x)

    def export(self, path, filename) -> None:
        self.to("cpu")
        layer = self.actor[0]
        while type(layer).__name__ == "Sequential":
            layer = layer[0]
        obs = torch.zeros(1, layer.in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={}, 
        )
