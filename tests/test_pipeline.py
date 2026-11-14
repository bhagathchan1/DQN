import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dqn_project.config import load_config
from dqn_project.pipeline import run_pipeline


CONFIG = Path(__file__).resolve().parents[1] / 'configs/default.yaml'


class PipelineTests(unittest.TestCase):
    def test_isolated_runs_and_held_out_seeds(self):
        def stage(module, arguments, directory, name):
            config = load_config(directory / 'config.yaml')
            self.assertEqual(config['evaluation']['seed'], config['run']['seed'] + 10000)
            if name == 'train':
                target = directory / 'checkpoints' / (config['run']['id'] + '_final.pt')
                target.parent.mkdir()
                target.touch()
            elif name in {'dqn', 'random'}:
                Path(arguments[-1]).write_text(json.dumps(dict(
                    policy=name, episodes=2, reward_mean=1, reward_std=0,
                    reward_min=1, reward_max=1)))
        with tempfile.TemporaryDirectory() as temp, patch('dqn_project.pipeline.run_stage', side_effect=stage):
            output = run_pipeline(str(CONFIG), str(Path(temp) / 'run'), [1, 2], smoke=True)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['runs']), 2)
            self.assertEqual(len((output / 'summary.csv').read_text().splitlines()), 5)
            with self.assertRaises(FileExistsError):
                run_pipeline(str(CONFIG), str(output))

    def test_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp, patch('dqn_project.pipeline.run_stage', side_effect=RuntimeError('failed stage')):
            output = Path(temp) / 'failed'
            with self.assertRaisesRegex(RuntimeError, 'failed stage'):
                run_pipeline(str(CONFIG), str(output))
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('failed stage', manifest['error'])

    def test_invalid_config_rejected(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.yaml'
            for section, key, value in [('training', 'train_frequency', 0),
                                        ('replay', 'capacity', 1),
                                        ('run', 'id', '../escape')]:
                original = config[section][key]
                config[section][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_config(path)
                config[section][key] = original
