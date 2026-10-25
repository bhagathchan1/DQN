"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    required = {"run", "environment", "training", "replay", "evaluation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    for section in required:
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(config['run'].get('id', ''))):
        raise ValueError('run.id must contain only letters, digits, underscores and hyphens')
    def integer(section: str, key: str, minimum: int = 1) -> int:
        value = config[section].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{section}.{key} must be an integer >= {minimum}')
        return value

    seed = integer('run', 'seed', 0)
    if seed >= 2**32:
        raise ValueError('run.seed must be below 2**32')
    for key in ['total_environment_steps', 'batch_size', 'train_frequency',
                'target_update_frequency', 'epsilon_decay_steps']:
        integer('training', key)
    integer('training', 'learning_starts', 0)
    integer('training', 'checkpoint_frequency', 0)
    if 'log_frequency' in config['training']:
        integer('training', 'log_frequency')
    for key in ['checkpoint_keep', 'validation_frequency']:
        if key in config['training']:
            integer('training', key, 0)
    if 'noop_max' in config['environment']:
        integer('environment', 'noop_max', 0)
    integer('replay', 'capacity')
    integer('evaluation', 'episodes')
    if 'max_steps_per_episode' in config['evaluation']:
        integer('evaluation', 'max_steps_per_episode')
    if 'seed' in config['evaluation']:
        integer('evaluation', 'seed', 0)
    if config['replay']['capacity'] < config['training']['batch_size']:
        raise ValueError('replay.capacity must be >= training.batch_size')
    storage = config['replay'].get('storage', 'stacks')
    if storage not in {'frames', 'stacks'}:
        raise ValueError('replay.storage must be frames or stacks')
    if storage == 'frames':
        if config['replay'].get('type') != 'uniform':
            raise ValueError('frame storage currently supports uniform replay only')
        if config['replay']['capacity'] < max(8, config['training']['batch_size'] + 4):
            raise ValueError('frame replay capacity must be >= batch_size + 4 and >= 8')
    if config['training'].get('optimizer', 'rmsprop') not in {'rmsprop', 'nature_rmsprop'}:
        raise ValueError('unknown training.optimizer')
    if config['replay'].get('type') not in {'uniform', 'prioritized'}:
        raise ValueError('replay.type must be uniform or prioritized')
    if config['replay']['type'] == 'prioritized':
        for key in ['alpha', 'beta_start']:
            value = config['replay'].get(key)
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f'replay.{key} must be in [0, 1]')
        value = config['replay'].get('priority_epsilon')
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('replay.priority_epsilon must be finite and positive')
    for section, keys in [('training', ['gamma', 'epsilon_start', 'epsilon_end']),
                          ('evaluation', ['epsilon'])]:
        for key in keys:
            value = config[section].get(key)
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f'{section}.{key} must be in [0, 1]')
    rate = config['training'].get('learning_rate')
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('training.learning_rate must be finite and positive')
    if not isinstance(config['environment'].get('id'), str) or not config['environment']['id']:
        raise ValueError('environment.id must be a nonempty string')
    return config
