"""Upload DeepQ Learning Lab to a Hugging Face Docker Space.

Only what the container needs is uploaded (Dockerfile, packaging files, src/,
trained_models/ and the web sources); experiments, results and notebooks stay
local. The Space then builds the Dockerfile itself.

One-time setup (your own account and token; nothing is stored in this repo):
    pip install huggingface_hub
    hf auth login                      # paste a token with "write" access

Deploy (creates the Space on the first run, updates it afterwards):
    python scripts/deploy_hf_space.py <your-hf-username>/deepq-learning-lab
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FILES = ['Dockerfile', '.dockerignore', 'pyproject.toml']
FOLDERS = ['src', 'trained_models', 'web']
SKIP_PARTS = {'node_modules', 'dist', '__pycache__'}

SPACE_CARD = """---
title: DeepQ Learning Lab
emoji: 🕹️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Play Pong and Breakout against Deep Q-Network agents
---

# DeepQ Learning Lab

Play Atari Pong and Breakout against Deep Q-Network (DQN) agents, and learn how
they were trained, step by step.

- **Pong · live duel:** you control the right paddle against a DQN that keeps
  learning from every 30-second match.
- **Breakout:** take turns with a trained DQN on the same level.
- **How it learns:** an interactive, beginner-friendly guide with the real
  training data and curves.

One game runs at a time, so if someone else is playing, wait for their match to finish.
This Space runs on CPU; what Pong learns online is reset whenever the Space restarts.

Developed by Yashas H M, Prabhakar Saurabh and Madipally Bhagath Chandra.
"""


def files_to_upload() -> list[Path]:
    paths = [PROJECT / name for name in FILES]
    for folder in FOLDERS:
        for path in sorted((PROJECT / folder).rglob('*')):
            relative = path.relative_to(PROJECT)
            if path.is_file() and not SKIP_PARTS & set(relative.parts) and not any(part.endswith(('.egg-info', '.tsbuildinfo')) for part in relative.parts):
                paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('space', help='Space ID, for example your-name/deepq-learning-lab')
    parser.add_argument('--private', action='store_true', help='Create the Space as private (first run only)')
    parser.add_argument('--dry-run', action='store_true', help='List what would be uploaded and stop')
    args = parser.parse_args()

    paths = files_to_upload()
    size = sum(path.stat().st_size for path in paths)
    print(f'{len(paths)} files, {size / 1e6:.0f} MB, plus the Space README')
    if args.dry_run:
        for path in paths:
            print(' ', path.relative_to(PROJECT).as_posix())
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(args.space, repo_type='space', space_sdk='docker', private=args.private, exist_ok=True)
    operations = [CommitOperationAdd(path.relative_to(PROJECT).as_posix(), str(path)) for path in paths]
    # The Space's own README carries its settings (Docker SDK, port 7860).
    operations.append(CommitOperationAdd('README.md', SPACE_CARD.encode('utf-8')))
    api.create_commit(args.space, repo_type='space', operations=operations,
                      commit_message='Deploy DeepQ Learning Lab')
    print(f'Uploaded. The Space is building: https://huggingface.co/spaces/{args.space}')


if __name__ == '__main__':
    main()
