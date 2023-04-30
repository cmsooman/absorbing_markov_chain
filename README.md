### Class for discrete time Markov processes with finite state space.

An absorbing Markov chain is a Markov chain in which every state can reach an absorbing state. An absorbing state is a state that, once entered, cannot be left. States which are not absorbing are termed transient states.

The probability of transitioning from state $i$ to state $j$ is denoted by $p_{i,j}$ and can be estimated in different ways, for example:
- Maximum-likelihood;
- Lagrangian.

The transition matrix $P$ with elements representing the probability of transition from state $i$ to state $j$ is:

$$P=\left\lbrack \matrix{
    p_{1,1} & p_{1,2} & \cdots & p_{1,N} \cr
    p_{2,1} & p_{2,2} & \cdots & p_{2,N} \cr
    \vdots & \vdots & \ddots & \vdots \cr 
    p_{N,1} & p_{N,2} & \cdots & p_{N,N}
} \right\rbrack$$ 

# Example

A person enters a casino with £3,000 and decides to gamble £1,000 a time at Black Jack. The person's strategy is to keep playin until the go bust or have £5,000. The probability of winning at Black Jack is 40% (0.4). The state space (using k to denote thousands) is 

$$S = \{0, £1k, £2k, £3k, £4k, $5k \}$$.

The absorbing states are $0$ and £5k$. So $p_{0,0} = 1$ and $p_{5,5} = 1$. The matrix $P$ is

$$P=\left\lbrack \matrix{
    1 & 0 & 0 & 0 & 0 & 0 \cr
   0.6 & 0 & 0.4 & 0 & 0 & 0 \cr
    0 & 0.6 & 0 & 0.4 & 0 & 0 \cr
    0 & 0 & 0.6 & 0 & 0.4 & 0 \cr
    0 & 0 & 0 & 0.6 & 0 & 0.4 \cr
    0 & 0 & 0 & 0 & 0 & 1 
}\right\rbrack$$  

## Canonical form

The transition matrix $P$ can be written in canonical form:

$$P=\left\lbrack \matrix{Q_{k \times k} & R_{k \times l} \cr 0_{l \times k} & I_{l \times l}} \right\rbrack,$$

where $Q_{k \times k}$ is the matrix of transition probabilities between transient states, $R_{k \times l}$ is the matrix of transition probabilities from transient states to absorbing states, $0_{l \times k}$ is the matrix of 0s representing transition probabilities from absorbing states to transient states, and $I_{l \times l}$ is the $l \times l$ identity matrix.  
