import numpy as np
from PIL import Image


def proc_obs(obs):
    image = Image.fromarray(obs, 'RGB').convert('L').resize((84, 110))
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                               image.size[0])


def get_successor_state(current, obs):
    return np.append(current[1:], [obs], axis=0)
