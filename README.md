### Class for discrete time Markov processes with finite state space.

An absorbing Markov chain is a Markov chain in which every state can reach an absorbing state. An absorbing state is a state that, once entered, cannot be left. States which are not absorbing is termed transient states.

The transition matrix $P$ with elements representing the probability of transition from state $i$ to state $j$ can be written in canonical form:

$ P = \begin{bmatrix} Q & R \\ 0 & I_r \end{bmatrix} $ 
