"""Run isolated train -> checkpoint evaluation -> random baseline experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_config


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_stage(module: str, arguments: list[str], directory: Path, name: str) -> None:
    """Subprocess cwd isolates legacy relative output paths; never change parent cwd."""
    command = [sys.executable, '-m', module, *arguments]
    print(f'[{directory.name}] {name}', flush=True)
    environment = os.environ.copy()
    environment.setdefault('OMP_NUM_THREADS', '1')
    environment.setdefault('MKL_NUM_THREADS', '1')
    with (directory / f'{name}.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=directory, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run_pipeline(config_path: str, output: str, seeds: list[int] | None = None,
                 smoke: bool = False, device: str | None = None) -> Path:
    config = load_config(config_path)
    seeds = seeds if seeds is not None else [int(config['run']['seed'])]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 - 10000 for s in seeds):
        raise ValueError('seeds must be unique nonnegative integers below 2**32 - 10000')
    output_path = Path(output).resolve()
    # Exclusive creation prevents overwriting or mixing old and new results.
    output_path.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_config': str(Path(config_path).resolve()), 'smoke': smoke,
        'python': sys.version, 'platform': platform.platform(), 'runs': [],
    }
    repo = Path(__file__).resolve().parents[2]
    try:
        manifest['git_revision'] = subprocess.check_output(
            ['git', '-c', f'safe.directory={repo.as_posix()}', 'rev-parse', 'HEAD'],
            cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        manifest['git_dirty'] = bool(subprocess.check_output(
            ['git', '-c', f'safe.directory={repo.as_posix()}', 'status', '--porcelain'],
            cwd=repo, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        manifest['git_revision'] = 'unknown'
    write_json(output_path / 'manifest.json', manifest)
    rows = []
    try:
        packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (output_path / 'requirements-lock.txt').write_text(packages, encoding='utf-8')
        for seed in seeds:
            current = copy.deepcopy(config)
            run_id = f"{current['replay']['type']}_seed{seed}"
            current['run'].update(id=run_id, seed=seed)
            current['evaluation']['seed'] = seed + 10000
            if device:
                current['training']['device'] = device
            if smoke:
                current['training'].update(total_environment_steps=64, learning_starts=16,
                                           batch_size=8, checkpoint_frequency=0,
                                           target_update_frequency=2, epsilon_decay_steps=64)
                current['replay']['capacity'] = 64
                current['evaluation'].update(episodes=2, max_steps_per_episode=100)
            directory = output_path / run_id
            directory.mkdir()
            resolved = directory / 'config.yaml'
            resolved.write_text(yaml.safe_dump(current, sort_keys=False), encoding='utf-8')
            record = {'id': run_id, 'training_seed': seed,
                      'evaluation_seed': seed + 10000, 'status': 'training'}
            manifest['runs'].append(record)
            write_json(output_path / 'manifest.json', manifest)
            run_stage('dqn_project.train', ['--config', str(resolved)], directory, 'train')
            checkpoint = directory / 'checkpoints' / f'{run_id}_final.pt'
            if not checkpoint.is_file():
                raise RuntimeError(f'training did not create {checkpoint}')
            record['status'] = 'evaluating'
            write_json(output_path / 'manifest.json', manifest)
            for policy, arguments in [('dqn', ['--checkpoint', str(checkpoint)]),
                                      ('random', ['--random-baseline'])]:
                result_path = directory / f'{policy}_evaluation.json'
                run_stage('dqn_project.evaluate', ['--config', str(resolved), *arguments,
                          '--output', str(result_path)], directory, policy)
                result = json.loads(result_path.read_text(encoding='utf-8'))
                rows.append({'run_id': run_id, 'training_seed': seed,
                             'evaluation_seed': seed + 10000,
                             **{key: result[key] for key in ['policy', 'episodes',
                                'reward_mean', 'reward_std', 'reward_min', 'reward_max']}})
            metrics = directory / 'results' / 'metrics' / f'{run_id}.csv'
            # Short smoke runs may not finish any training episodes.
            if metrics.is_file():
                with metrics.open(encoding='utf-8') as stream:
                    has_episodes = any(row.get('event') == 'episode' for row in csv.DictReader(stream))
                if has_episodes:
                    run_stage('dqn_project.plot_results', [str(metrics), '--output',
                              str(directory / 'training_reward.png')], directory, 'plot')
            record['status'] = 'complete'
            with (output_path / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(output_path / 'manifest.json', manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output_path / 'manifest.json', manifest)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output', required=True, help='New experiment directory (must not exist)')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--smoke', action='store_true', help='64 decisions; validates plumbing, not learning')
    args = parser.parse_args()
    print(run_pipeline(args.config, args.output, args.seeds, args.smoke, args.device))


if __name__ == '__main__':
    main()
