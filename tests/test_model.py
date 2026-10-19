import unittest

import torch

from dqn_project.model import DQN


class ModelTests(unittest.TestCase):
    def test_output_shape_and_finite_values(self) -> None:
        model = DQN(action_count=6)
        output = model(torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_shape_is_rejected(self) -> None:
        model = DQN(action_count=2)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 84, 84))


if __name__ == "__main__":
    unittest.main()
