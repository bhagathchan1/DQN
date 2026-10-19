"""Reset and step an installed visual Gymnasium environment."""

import argparse

from dqn_project.environment import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="ALE/Breakout-v5")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    env = make_env(args.env)
    try:
        state, info = env.reset(seed=args.seed)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, step_info = env.step(action)
        print(
            {
                "environment": args.env,
                "action_count": env.action_space.n,
                "state_shape": state.shape,
                "state_dtype": str(state.dtype),
                "next_state_shape": next_state.shape,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "reset_info_keys": sorted(info),
                "step_info_keys": sorted(step_info),
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
