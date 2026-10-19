import unittest

import numpy as np

from dqn_project.environment import DQNEnv, preprocess_frame


class FakeActionSpace:
    n = 3


class FakeEnv:
    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.step_count = 0
        self.closed = False

    def reset(self, **kwargs):
        self.step_count = 0
        return np.full((20, 30, 3), 20, dtype=np.uint8), {"seed": kwargs.get("seed")}

    def step(self, action):
        self.step_count += 1
        frame = np.full((20, 30, 3), self.step_count * 30, dtype=np.uint8)
        return frame, 0.5, self.step_count == 2, False, {"action": action}

    def close(self):
        self.closed = True


class EnvironmentTests(unittest.TestCase):
    def test_preprocess_shape_dtype_and_luminance(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[..., 0] = 255
        result = preprocess_frame(frame)
        self.assertEqual(result.shape, (84, 84))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(70 <= int(result.mean()) <= 80)

    def test_stack_action_repeat_and_termination(self) -> None:
        base = FakeEnv()
        env = DQNEnv(base, frame_stack=4, action_repeat=4)
        initial, info = env.reset(seed=7)
        self.assertEqual(initial.shape, (4, 84, 84))
        self.assertEqual(info["seed"], 7)
        next_state, reward, terminated, truncated, info = env.step(2)
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], 2)
        self.assertGreater(int(next_state[-1].mean()), int(initial[-1].mean()))
        env.close()
        self.assertTrue(base.closed)


if __name__ == "__main__":
    unittest.main()
