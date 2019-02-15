from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


def build_model(input_shape, actions):
    model = Sequential()
    model.add(Conv2D(32, 8, strides=(4, 4),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_first'))
    model.add(Conv2D(64, 4, strides=(2, 2),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_first'))
    model.add(Conv2D(64, 3, strides=(1, 1),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions))
    return model
