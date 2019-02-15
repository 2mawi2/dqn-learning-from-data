from random import randrange

import numpy as np


class ReplayMemory:
    def __init__(self, mini_batch_size, replay_memory_size, logger) -> None:

        self.logger = logger
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.experiences = []

    def get_random_state(self):

        return self.experiences[randrange(0, len(self.experiences))]['source']

    def add_experience(self, source, action, reward, dest, final):

        if len(self.experiences) >= self.replay_memory_size:
            self.experiences.pop(0)
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})
        if (len(self.experiences) % 100 == 0) and (len(self.experiences) < self.replay_memory_size) and (
                self.logger is not None):
            self.logger.log("Collected %d samples of %d" %
                            (len(self.experiences), self.replay_memory_size))

    def get_sample_batch(self):

        batch = []
        for i in range(self.mini_batch_size):
            batch.append(self.experiences[randrange(0, len(self.experiences))])
        return np.asarray(batch)
