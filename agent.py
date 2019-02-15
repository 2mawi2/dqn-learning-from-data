import numpy as np

from memory import ReplayMemory
from network import NeuralNet
from random import random, randint


class Agent:
    def __init__(self,
                 actions,
                 network_input_shape,
                 replay_memory_size=1024,
                 mini_batch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.9,
                 dropout_prob=0.1,
                 epsilon=1,
                 epsilon_decrease_rate=0.99,
                 min_epsilon=0.1,
                 load_path=None,
                 logger=None):

        self.network_input_shape = network_input_shape
        self.actions = actions
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.load_path = load_path
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decrease_rate = epsilon_decrease_rate
        self.min_epsilon = min_epsilon
        self.logger = logger
        self.mini_batch_size = mini_batch_size
        self.training_count = 0

        self.DQN = self.create_neural_net()
        self.DQN_target = self.create_neural_net()
        self.DQN_target.model.set_weights(self.DQN.model.get_weights())
        self.replay_memory = ReplayMemory(mini_batch_size, replay_memory_size, logger)

    def create_neural_net(self):
        return NeuralNet(
            self.actions,
            self.network_input_shape,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            mini_batch_size=self.mini_batch_size,
            dropout_prob=self.dropout_prob,
            load_path=self.load_path,
            logger=self.logger
        )

    def get_action(self, state, testing=False, force_random=False):
        is_random = (random() < (self.epsilon if not testing else 0.05))
        if force_random or is_random:
            return randint(0, self.actions - 1)
        else:
            q_values = self.DQN.predict(state)
            return np.argmax(q_values)

    def get_max_q(self, state):
        q_values = self.DQN.predict(state)
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        return np.random.choice(idxs)

    def get_random_state(self):
        return self.replay_memory.get_random_state()

    def get_experience_size(self):
        return len(self.replay_memory.experiences)

    def add_experience(self, source, action, reward, dest, final):
        self.replay_memory.add_experience(source, action, reward, dest, final)

    def train(self):
        self.training_count += 1
        print(f'Training session #{self.training_count:d} - epsilon: {self.epsilon:f}')
        batch = self.replay_memory.get_sample_batch()
        self.DQN.train(batch, self.DQN_target)

    def decay_epsilon(self):
        if self.epsilon - self.epsilon_decrease_rate > self.min_epsilon:
            self.epsilon -= self.epsilon_decrease_rate
        else:
            self.epsilon = self.min_epsilon

    def update_target_network(self):
        if self.logger is not None:
            self.logger.log('Updating target network...')
        self.DQN_target.model.set_weights(self.DQN.model.get_weights())

    def quit(self):
        if self.load_path is None:
            if self.logger is not None:
                self.logger.log('exiting...')
            self.DQN.save(append='_DQN')
            self.DQN_target.save(append='_DQN_target')
