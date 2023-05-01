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

#### Example

A person enters a casino with £30 and decides to gamble £10 a time at Black Jack. The person's strategy is to keep playing until the go bust or have £50. The probability of winning at Black Jack is 40% (0.4). The state space is 

$$S = \lbrace 0, £10, £20, £30, £40, £50 \rbrace.$$

The absorbing states are $0$ and $£50$. So $p_{0,0} = 1$ and $p_{5,5} = 1$. The matrix $P$ is

$$P=\left\lbrack \matrix{
    1 & 0 & 0 & 0 & 0 & 0 \cr
   0.6 & 0 & 0.4 & 0 & 0 & 0 \cr
    0 & 0.6 & 0 & 0.4 & 0 & 0 \cr
    0 & 0 & 0.6 & 0 & 0.4 & 0 \cr
    0 & 0 & 0 & 0.6 & 0 & 0.4 \cr
    0 & 0 & 0 & 0 & 0 & 1 
}\right\rbrack$$  

#### Canonical form

The transition matrix $P$ can be written in canonical form:

$$P=\left\lbrack \matrix{Q_{k \times k} & R_{k \times l} \cr 0_{l \times k} & I_{l \times l}} \right\rbrack,$$

where $Q_{k \times k}$ is the matrix of transition probabilities between transient states, $R_{k \times l}$ is the matrix of transition probabilities from transient states to absorbing states, $0_{l \times k}$ is the matrix of 0s representing transition probabilities from absorbing states to transient states, and $I_{l \times l}$ is the $l \times l$ identity matrix.  

In the case of our above example with the gambler in the casino, we would rewrite the matrix P accordingly:

$$P=\left\lbrack \matrix{
    p_{1,1} & p_{1,2} & p_{1,3} & p_{1,4} & p_{1,0} & p_{1,5} \cr
    p_{2,1} & p_{2,2} & p_{2,3} & p_{2,4} & p_{2,0} & p_{2,5} \cr
    p_{3,1} & p_{3,2} & p_{3,3} & p_{3,4} & p_{3,0} & p_{3,5} \cr
    p_{4,1} & p_{4,2} & p_{4,3} & p_{4,4} & p_{4,0} & p_{4,5} \cr
    p_{0,1} & p_{0,2} & p_{0,3} & p_{0,4} & p_{0,0} & p_{0,5} \cr
    p_{5,1} & p_{5,2} & p_{5,3} & p_{5,4} & p_{5,0} & p_{5,5} \cr
}\right\rbrack
= \left\lbrack \matrix{
    0 & 0.4 & 0 & 0 & 0.6 & 0 \cr
    0.6 & 0 & 0.4 & 0 & 0 & 0 \cr
    0 & 0.6 & 0 & 0.4 & 0 & 0 \cr
    0 & 0 & 0.6 & 0 & 0 & 0.4 \cr
    0 & 0 & 0 & 0 & 1 & 0 \cr
    0 & 0 & 0 & 0 & 0 & 1 
}\right\rbrack
 $$

The different components of the canonical matrix are:

$$
\begin{aligned}
Q_{4\times4} &= \left\lbrack \matrix{
    0 & 0.4 & 0 & 0 \cr
    0.6 & 0 & 0.4 & 0 \cr
    0 & 0.6 & 0 & 0.4 \cr
    0 & 0 & 0.6 & 0
}\right\rbrack,\\
R_{4\times2} &= \left\lbrack \matrix{
    0.6 & 0 \cr
    0 & 0 \cr
    0 & 0 \cr
    0 & 0.4
}\right\rbrack,\\
0_{2\times4} &= \left\lbrack \matrix{
    0 & 0 & 0 & 0 \cr
    0 & 0 & 0 & 0
}\right\rbrack,\\
I_{2\times2} &= \left\lbrack \matrix{
    1 & 0 \cr
    0 & 1
}\right\rbrack.
\end{aligned}
$$

#### Fundamental matrix

A property of an absorbing Markov chain is the expected number of visits to a transient state $j$ starting at a trnasient state $i$ before being absorbed. We could obtain this by summing the matrix $Q_{k \times k}$ from $0$ to $\infty$. The resulting matrix is called the fundamental matrix:

$$
F:=\sum_{m=0}^{\infty}Q^m = \left(I_{k \times k} - Q_{k \times k} \right)^{-1} 
$$  
