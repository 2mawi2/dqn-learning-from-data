import argparse
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('-v', '--video', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-e', '--environment', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--replay-memory-size', type=int, default=1e6)
    parser.add_argument('--target-network-update-freq', type=int, default=10e3)
    parser.add_argument('--avg-val-computation-freq', type=int, default=50e3)
    parser.add_argument('--discount-factor', type=float, default=0.99)
    parser.add_argument('--update-freq', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.00025)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--min-epsilon', type=float, default=0.1)
    parser.add_argument('--epsilon-decrease', type=float, default=9e-7)
    parser.add_argument('--replay-start-size', type=int, default=50e3)
    parser.add_argument('--initial-random-actions', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--max-episodes', type=int, default=np.inf)
    parser.add_argument('--max-episode-length', type=int, default=np.inf)
    parser.add_argument('--max-frames-number', type=int, default=50e6)
    parser.add_argument('--test-freq', type=int, default=250000)
    parser.add_argument('--validation-frames', type=int, default=135000)
    parser.add_argument('--test-states', type=int, default=30)
    return parser
