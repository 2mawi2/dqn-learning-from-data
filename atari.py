import os

import atexit
import gym
import numpy as np
import random
import utils
from agent import Agent
from evaluation import evaluate
from logger import Logger
from parser import get_parser


def exit_handler():
    global agent
    agent.quit()


def run():
    args = get_parser().parse_args()

    logger = Logger(debug=args.debug, append=args.environment)
    atexit.register(exit_handler)

    test_scores = []
    test_mean_q = []
    test_states = []

    env = gym.make(args.environment)
    network_input_shape = (4, 110, 84)
    agent = create_agent(args, env, logger, network_input_shape)

    logger.log({
        'Action space': env.action_space.n,
        'Observation space': env.observation_space.shape
    })
    logger.log(vars(args))
    training_csv = 'training_info.csv'
    eval_csv = 'evaluation_info.csv'
    test_csv = 'test_score_mean_q_info.csv'
    logger.to_csv(training_csv, 'length,score')
    logger.to_csv(eval_csv, 'length,score')
    logger.to_csv(test_csv, 'avg_score,avg_Q')

    episode = 0
    frame_counter = 0

    if args.train:
        while episode < args.max_episodes:
            logger.log("Episode %d" % episode)
            score = 0

            obs = utils.proc_obs(env.reset())

            current_state = np.array([obs, obs, obs, obs])

            t = 0
            frame_counter += 1
            while t < args.max_episode_length:
                if frame_counter > args.max_frames_number:
                    agent.quit()

                if args.video:
                    env.render()

                action = agent.get_action(np.asarray([current_state]))

                obs, reward, done, info = env.step(action)
                obs = utils.proc_obs(obs)
                next_state = utils.get_successor_state(current_state, obs)

                frame_counter += 1

                clipped_reward = np.clip(reward, -1, 1)
                agent.add_experience(np.asarray([current_state]),
                                     action,
                                     clipped_reward,
                                     np.asarray([next_state]),
                                     done)

                if t % args.update_freq == 0 and agent.get_experience_size() >= args.replay_start_size:
                    agent.train()
                    if agent.training_count % args.target_network_update_freq == 0 \
                            and agent.training_count >= args.target_network_update_freq:
                        agent.update_target_network()
                    if agent.training_count % args.avg_val_computation_freq == 0 \
                            and agent.training_count >= args.avg_val_computation_freq:
                        logger.to_csv(test_csv,
                                      [np.mean(test_scores), np.mean(test_mean_q)])
                        del test_scores[:]
                        del test_mean_q[:]

                if agent.get_experience_size() >= args.replay_start_size:
                    agent.decay_epsilon()

                current_state = next_state
                score += reward

                if done or t == args.max_episode_length - 1:
                    logger.to_csv(training_csv, [t, score])
                    logger.log(f"Length: {t + 1:d}; Score: {score:d}\n")
                    break

                t += 1

                if frame_counter % args.test_freq == 0:
                    t_evaluation, score_evaluation = evaluate(agent, args, logger)
                    logger.to_csv(eval_csv, [t_evaluation, score_evaluation])

                if len(test_states) < args.test_states:
                    for _ in range(random.randint(1, 5)):
                        test_states.append(agent.get_random_state())
                else:
                    test_scores.append(score)
                    test_q_values = [agent.get_max_q(state) for state in test_states]
                    test_mean_q.append(np.mean(test_q_values))

            episode += 1

    if args.eval:
        logger.log(evaluate(agent, args, logger))


def create_agent(args, env, logger, network_input_shape):
    return Agent(env.action_space.n,
                 network_input_shape,
                 replay_memory_size=args.replay_memory_size,
                 mini_batch_size=args.minibatch_size,
                 learning_rate=args.learning_rate,
                 discount_factor=args.discount_factor,
                 dropout_prob=args.dropout,
                 epsilon=args.epsilon,
                 epsilon_decrease_rate=args.epsilon_decrease,
                 min_epsilon=args.min_epsilon,
                 load_path=args.load,
                 logger=logger)


if __name__ == '__main__':
    run()
