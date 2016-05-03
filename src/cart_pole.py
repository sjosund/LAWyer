import logging
import os
from collections import deque, defaultdict

import numpy as np
import gym


class LAWyer(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_n = action_space.n
        self.history = deque()
        self.decay = 0.99
        self.state_action_values = defaultdict(
            lambda: 0.01 * np.random.randn(self.action_n)
        )

    def act(self, observation, reward, done):
        obs = tuple(observation.tolist())
        action = self.state_action_values[obs].argmax()# if np.random.rand() > 0.05 else np.random.choice(range(self.action_n))

        for i, (state, action_) in enumerate(self.history):
            self.state_action_values[state][action_] = reward * self.decay ** i
        self.history.appendleft((obs, action))

        print(self.state_action_values[obs].argmax(), action)
        return action


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    agent = LAWyer(env.action_space)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)

    episode_count = 100
    max_steps = 20000000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir, algorithm_id='random')