"""
Markov Decision Process - The Bellman equation adapted to Reinforcement Learning
R is the reward matrix for each state

Each line in the matrix in the example represents a letter from A to F, and each column
represents a letter from A to F. All possible states are represented. The `1` values represent
the nodes (vertices) of the graph
"""

import numpy as np

R: np.matrix = np.matrix([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 100, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
])

Q = np.matrix(np.zeros([6, 6]))
gamma: float = 0.8

# The agent starts in state 1, for example. You can start wherever you want because it's a random process.
# Note that only values > 0 are taken into account. They represent the possible moves (decisions)
agent_s_state = 1

# The possible "a" actions when the agent is in a particular state


def possible_actions(state):
    current_state_row = R[state,]
    possible_act = np.where(current_state_row > 0)[1]
    return possible_act


# get available actions in the current state
possible_actions_results = possible_actions(agent_s_state)


# This function chooses at random which action to be performed within the range
# of all the available actions.
def action_choice(available_actions_range):
    if (sum(available_actions_range) > 0):
        next_action = int(np.random.choice(available_actions_range, 1))
    if (sum(available_actions_range) <= 0):
        next_action = int(np.random.choice(5, 1))
    return next_action


# sample next action to be performed
action = action_choice(possible_actions_results)


# A version of Bellman's equation for reinforcement learning using the Q function
# This reinforcement algorithm is a memoryless process
# The transition function T from one state to another
# is not in the equation below.  T is done by the random choice above
def reward(current_state, action, gamma):
    max_state = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_state.shape[0] > 1:
        max_state = int(np.random.choice(max_state, size=1))
    else:
        max_state = int(max_state)

    max_value = Q[action, max_state]

    # Q Function
    Q[current_state, action] = R[current_state, action] + gamma * max_value


# Rewarding Q matrix
reward(agent_s_state, action, gamma)

# Learning over n iterations depending on the convergence of the system
# A convergence function can replace the systematic repeating of the process
# by comparing the sum of the Q matrix to that of Q matrix n-1 in the
# previous iteration
for i in range(50000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    PossibleAction = possible_actions(current_state)
    action = action_choice(PossibleAction)
    reward(current_state, action, gamma)

# Displaying Q before the norm of Q phase
print("Q  :")
print(Q)

# Norm of Q
print("Normed Q :")
print(Q/np.max(Q)*100)
