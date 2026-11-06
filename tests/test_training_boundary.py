# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual trainer's first-batch control flow without running CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class BatchReached(Exception):
    pass


class TrainingBoundaryTests(unittest.TestCase):
    def run_fit_until_first_batch(self, budget):
        source = Path(__file__).resolve().parents[1] / "verl/trainer/ray_trainer.py"
        module = ast.parse(source.read_text())
        trainer = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
        # Compile the unchanged method, substituting only its external GPU boundary.
        def first_batch(batch):
            raise BatchReached()
        namespace = {
            "Tracking": lambda **kwargs: SimpleNamespace(log=lambda **kw: None),
            "DataProto": SimpleNamespace(from_single_dict=first_batch),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        fake = SimpleNamespace(
            config=SimpleNamespace(
                trainer=SimpleNamespace(project_name="test", experiment_name="test", logger=["console"], total_episodes=1),
                to_dict=lambda: {},
            ),
            training_steps=budget, train_dataloader=[{}], val_reward_fn=None,
            _load_checkpoint=lambda: None, _save_checkpoint=lambda: None,
        )
        namespace["fit"](fake)
        return fake.global_steps

    def test_one_step_reaches_a_training_batch(self):
        with self.assertRaises(BatchReached):
            self.run_fit_until_first_batch(1)

    def test_zero_budget_does_not_increment_or_train(self):
        self.assertEqual(self.run_fit_until_first_batch(0), 0)


if __name__ == "__main__":
    unittest.main()
