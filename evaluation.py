from __future__ import print_function
import gym
import numpy as np
import utils

max_mean_score = 0


def evaluate(agent, args, logger):
    global max_mean_score

    evaluation_csv = 'evaluation.csv'
    logger.to_csv(evaluation_csv, 'length,score')
    env = gym.make(args.environment)
    scores = list()
    frame_counter = 0

    while frame_counter < args.validation_frames:

        remaining_random_actions = args.initial_random_actions
        obs = utils.proc_obs(env.reset())

        frame_counter += 1
        current_state = np.array([obs, obs, obs, obs])
        t = 0
        episode = 0
        score = 0

        while True:

            if args.video:
                env.render()

            action = agent.get_action(np.asarray([current_state]),
                                      testing=True,
                                      force_random=remaining_random_actions > 0)
            obs, reward, done, info = env.step(action)
            obs = utils.proc_obs(obs)
            current_state = utils.get_successor_state(current_state, obs)

            if remaining_random_actions > 0:
                remaining_random_actions -= 1

            score += reward
            t += 1
            frame_counter += 1

            if done or t > args.max_episode_length:
                episode += 1
                print(f'Episode {episode:d} end\n---------------\nFrame counter: {frame_counter:d}\n')
                print(f'Length: {t:d}\n, Score: {score:f}\n\n')
                logger.to_csv(evaluation_csv, [t, score])
                break

        scores.append([t, score])

    scores = np.asarray(scores)
    max_indices = np.argwhere(scores[:, 1] == np.max(scores[:, 1])).ravel()
    max_idx = np.random.choice(max_indices)

    if max_mean_score < np.mean(scores):
        max_mean_score = np.mean(scores)
        agent.DQN.save(append='_best')

    return scores[max_idx, :].ravel()
