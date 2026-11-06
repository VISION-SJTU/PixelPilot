# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the actual configuration schema without importing GPU workers."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


class ConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("omegaconf"), "OmegaConf is needed for schema validation")
    def test_grpo_yaml_merges_with_retained_trainer_schema(self):
        root = Path(__file__).resolve().parents[1]
        code = '''
import importlib, sys, types
from pathlib import Path
from omegaconf import OmegaConf
root = Path.cwd()
# Bypass only package initializers that import torch; load the actual config files.
for name in ["verl", "verl.workers", "verl.workers.actor", "verl.workers.critic",
             "verl.workers.reward", "verl.workers.rollout", "verl.trainer"]:
    module = types.ModuleType(name)
    module.__path__ = [str(root / name.replace(".", "/"))]
    sys.modules[name] = module
for name, attrs in [("actor", ["ActorConfig", "FSDPConfig", "ModelConfig", "OptimConfig", "RefConfig"]),
                    ("critic", ["CriticConfig"]), ("reward", ["RewardConfig"]), ("rollout", ["RolloutConfig"])]:
    module = importlib.import_module(f"verl.workers.{name}.config")
    for attr in attrs:
        setattr(sys.modules[f"verl.workers.{name}"], attr, getattr(module, attr))
from verl.trainer.config import PPOConfig
config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load("configs/grpo_7b.yaml")))
config.deep_post_init()
assert config.worker.rollout.n == 8
assert config.worker.rollout.response_length == 4096
assert config.worker.reward.compute_score == "nuscenes"
assert config.data.val_files is None
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
