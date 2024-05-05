import copy
import random
import numpy as np

class RiverSwim:
    def __init__(self, current, seed=1234):
        self.num_states = 6
        self.num_actions = 2  # O <=> LEFT, 1 <=> RIGHT

        # Larger current makes it harder to swim up the river
        self.currents = ['WEAK', 'MEDIUM', 'STRONG']
        assert current in self.currents
        self.current = self.currents.index(current) + 1
        assert self.current in [1, 2, 3]

        # Configure reward function
        R = np.zeros((self.num_states, self.num_actions))
        R[0, 0] = 0.005
        R[5, 1] = 1.

        # Configure transition function
        T = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Encode initial and rewarding state transitions
        T[0, 0, 0] = 1.
        T[0, 1, 0] = 0.6
        T[0, 1, 1] = 0.4

        T[5, 1, 5] = 0.6
        T[5, 1, 4] = 0.4
        T[5, 0, 4] = 1.

        # Encode intermediate state transitions
        for s in range(1, self.num_states - 1):
            left, right = 0, 1

            # Going left always succeeds
            T[s, left, s - 1] = 1.

            # Going right sometimes succeeds
            T[s, right, s] = 0.6
            T[s, right, s - 1] = 0.09 * self.current
            T[s, right, s + 1] = 0.4 - T[s, right, s - 1]
            assert np.isclose(np.sum(T[s, right]), 1.)

        self.R = np.array(R)
        self.T = np.array(T)

        # Agent always starts at the opposite end of the river
        self.init_state = 0
        self.curr_state = self.init_state

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_model(self):
        return copy.deepcopy(self.R), copy.deepcopy(self.T)

    def reset(self):
        return self.init_state

    def step(self, action):
        reward = self.R[self.curr_state, action]
        next_state = np.random.choice(range(self.num_states), p=self.T[self.curr_state, action])
        self.curr_state = next_state
        return reward, next_state