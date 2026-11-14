import unittest

import torch

from dqn_project.model import DQN


class ModelTests(unittest.TestCase):
    def test_dark_uint8_matches_normalized_float(self) -> None:
        model = DQN(action_count=2)
        pixels = torch.ones((1, 4, 84, 84), dtype=torch.uint8)
        torch.testing.assert_close(model(pixels), model(pixels.float() / 255.0))

    def test_output_shape_and_finite_values(self) -> None:
        model = DQN(action_count=6)
        output = model(torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_shape_is_rejected(self) -> None:
        model = DQN(action_count=2)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 84, 84))

    def test_nature_checkpoint_layer_names_are_unchanged(self) -> None:
        names = set(DQN(action_count=6).state_dict())
        self.assertIn("head.3.weight", names)
        self.assertFalse(any(name.startswith(("value.", "advantage.")) for name in names))

    def test_dueling_combines_value_and_mean_centred_advantage(self) -> None:
        torch.manual_seed(3)
        model = DQN(action_count=6, architecture="dueling")
        states = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
        output = model(states)
        self.assertEqual(tuple(output.shape), (3, 6))
        self.assertTrue(torch.isfinite(output).all())
        features = model.features(states.float() / 255.0)
        value, advantage = model.value(features), model.advantage(features)
        torch.testing.assert_close(output, value + advantage - advantage.mean(dim=1, keepdim=True))
        # The mean over actions of Q equals V(s): the advantage stream is centred.
        torch.testing.assert_close(output.mean(dim=1, keepdim=True), value)

    def test_unknown_architecture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DQN(action_count=2, architecture="transformer")


if __name__ == "__main__":
    unittest.main()
