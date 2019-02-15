import numpy as np

from model import build_model


class NeuralNet:
    def __init__(self, actions, input_shape,
                 mini_batch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 dropout_prob=0.1,
                 load_path=None,
                 logger=None):

        self.actions = actions
        self.discount_factor = discount_factor
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.logger = logger
        self.train_hist_csv = 'train_hist.csv'

        if self.logger is not None:
            self.logger.to_csv(self.train_hist_csv, 'Loss,Accuracy')

        self.model = build_model(input_shape, self.actions)

        if load_path is not None:
            self.load(load_path)

        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, batch, target_network):

        x_train = []
        y_train = []

        for observation in batch:
            x_train.append(observation['source'].astype(np.float64))

            next_state = observation['dest'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            t = list(self.predict(observation['source'])[0])
            if observation['final']:
                t[observation['action']] = observation['reward']
            else:
                t[observation['action']] = observation['reward'] + \
                                           self.discount_factor * next_q_value
            y_train.append(t)

        x_train = np.asarray(x_train).squeeze()
        y_train = np.asarray(y_train).squeeze()

        h = self.model.fit(x_train,  # states
                           y_train,  # targets
                           batch_size=self.mini_batch_size,
                           nb_epoch=1)

        if self.logger is not None:
            self.logger.to_csv(self.train_hist_csv,
                               [h.history['loss'][0], h.history['acc'][0]])

    def predict(self, state):

        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=''):
        f = ('model%s.h5' % append) if filename is None else filename

        if self.logger is not None:
            self.logger.log(f'Saving model {f}')

        self.model.save_weights(self.logger.path + f)

    def load(self, path):

        if self.logger is not None:
            self.logger.log('Loading weights...')
        self.model.load_weights(path)
