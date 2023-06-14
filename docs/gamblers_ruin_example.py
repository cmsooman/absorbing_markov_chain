import numpy as np
from absorbing_markov_chain.absorbing_markov_chain import AbsorbingMarkovChain

P = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0.6, 0, 0.4, 0, 0, 0],
        [0, 0.6, 0, 0.4, 0, 0],
        [0, 0, 0.6, 0, 0.4, 0],
        [0, 0, 0, 0.6, 0, 0.4],
        [0, 0, 0, 0, 0, 1],
    ]
)  # the transition matrix for the problem

P_c = np.array(
    [
        [0, 0.4, 0, 0, 0.6, 0],
        [0.6, 0, 0.4, 0, 0, 0],
        [0, 0.6, 0, 0.4, 0, 0],
        [0, 0, 0.6, 0, 0, 0.4],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)  # te transition matrix rearranged to Canonical form

# Determine the number of bets the gambler makes before the game is over

# First we create an object using te class. We know that there are 2 absorbing states and we can set the number of projected years to 1 for now
gamblers_ruin_problem = AbsorbingMarkovChain(2, P_c)

# First we calculate the fundamental matrix F
F = gamblers_ruin_problem.fundamental_matrix()
print(F)

# From the fundamental matrix F, we sum the third row to get the expected number of games before absorption from the state {£30}.
expected_num_games = gamblers_ruin_problem.absorb_times()
print(expected_num_games[2])

# We can also calculate the probabilty of the gambler facing financial ruin or reaching £50 when starting with £30
solution_matrix = gamblers_ruin_problem.absorb_probs()
print(solution_matrix[2])

# So the probabilty of the gambler starting with £30 facing financial ruin is 64% and the probability of reaching £50 is 36%.

# As a sense check, find the power n at which the matrix remains stable
long_run_matrix = gamblers_ruin_problem.forward_matrix(1)
print(long_run_matrix)

count = 1
current_matrix = gamblers_ruin_problem.forward_matrix(count)
updated_matrix = gamblers_ruin_problem.forward_matrix(count + 1)
while not np.allclose(
    current_matrix, updated_matrix, rtol=1e-05, atol=1e-08, equal_nan=False
):
    current_matrix = gamblers_ruin_problem.forward_matrix(count)
    updated_matrix = gamblers_ruin_problem.forward_matrix(count + 1)
    count += 1
print(count)
print(updated_matrix[0:4, 4 : updated_matrix.shape[1]])

# Notice how this is equal to the solution matrix
print(solution_matrix)

# Calculate the expected number of steps prior to being absorbed
avg_absorb_times = gamblers_ruin_problem.absorb_times()
print(avg_absorb_times)
