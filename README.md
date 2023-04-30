### Class for discrete time Markov processes with finite state space.

An absorbing Markov chain is a Markov chain in which every state can reach an absorbing state. An absorbing state is a state that, once entered, cannot be left. States which are not absorbing are termed transient states.

The probability of transitioning from state $i$ to state $j$ is denoted by $p_{i,j}$ and can be estimated in different ways, for example:
$$
\begin{itemize}
    \item Maximum-likelihood;
    \item Lagrangian.
\end{itemize}
$$

The transition matrix $P$ with elements representing the probability of transition from state $i$ to state $j$ is:

$$P=\left\lbrack \matrix{
    p_{1,1} & p_{1,2} & \cdots & p_{1,N} \cr
    p_{2,1} & p_{2,2} & \cdots & p_{2,N} \cr
    \ddots & \ddots & \vdots & \ddots \cr 
    p_{N,1} & p_{N,2} & \cdots & p_{N,N}
} \right\rbrack$$ 

## Canonical form

The transition matrix $P$ can be written in canonical form:

$$P=\left\lbrack \matrix{Q_{k \times k} & R_{k \times l} \cr 0_{l \times k} & I_{l \times l} \right\rbrack,$$

where $Q_{k \times k}$ is the matrix of transition probabilities between transient states, $R_{k \times l}$ is the matrix of transition probabilities from transient states to absorbing states, $0_{l \times k}$ is the matrix of 0s representing transition probabilities from absorbing states to transient states, and $I_{l \times l} $ is the $l \times l $ identity matrix.  
