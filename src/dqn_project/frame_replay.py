"""Uniform replay storing one processed frame per transition, reconstructing stacks."""
from pathlib import Path

import numpy as np

from .replay_buffer import ReplayBatch


class FrameReplayBuffer:
    def __init__(self, capacity: int, state_shape=(4, 84, 84), seed=None, directory=None):
        if capacity < 8 or state_shape != (4, 84, 84):
            raise ValueError('frame replay requires capacity >= 8 and shape (4,84,84)')
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        shape = (capacity, 84, 84)
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self.frames = np.memmap(path / 'frames.uint8', dtype=np.uint8, mode='w+', shape=shape)
        else:
            self.frames = np.empty(shape, dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.starts = {}
        self.count = 0
        self.pending_start = True

    def start_episode(self):
        self.pending_start = True

    def __len__(self):
        return min(self.count, self.capacity - 4)

    def add(self, state, action, reward, next_state, terminated):
        slot = self.count % self.capacity
        # Boundary stacks are retained only as long as their transitions survive.
        self.starts.pop(self.count - self.capacity, None)
        if self.pending_start:
            self.starts[self.count] = np.asarray(state, dtype=np.uint8).copy()
            self.pending_start = False
        self.frames[slot] = next_state[-1]
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminated[slot] = terminated
        self.count += 1
        return slot

    def _state(self, index):
        tail = []
        for offset in range(4):
            previous = index - offset
            if previous in self.starts:
                initial = self.starts[previous]
                return np.concatenate([initial, np.asarray(tail[::-1], dtype=np.uint8)], axis=0)[-4:] if tail else initial.copy()
            tail.append(self.frames[(previous - 1) % self.capacity])
        return np.stack(tail[::-1])

    def sample(self, batch_size, beta=None):
        if not 0 < batch_size <= len(self):
            raise ValueError('not enough sampleable transitions')
        oldest = max(0, self.count - self.capacity + 4)
        indices = self.rng.choice(self.count - oldest, size=batch_size, replace=True) + oldest
        slots = indices % self.capacity
        states = np.stack([self._state(int(i)) for i in indices])
        successors = np.concatenate([states[:, 1:], self.frames[slots, None]], axis=1)
        return ReplayBatch(states, self.actions[slots].copy(), self.rewards[slots].copy(),
                           successors, self.terminated[slots].copy(),
                           np.ones(batch_size, dtype=np.float32), slots)

    def update_priorities(self, indices, priorities):
        pass

    def close(self):
        if isinstance(self.frames, np.memmap):
            self.frames.flush()
            self.frames._mmap.close()
