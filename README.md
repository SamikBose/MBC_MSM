# MBC_MSM

**Methodological details - **

*Objective:* Building Markov state models with weighted ensemble (WE) simulation data.

*Problem:* When constructing a transition matrix $\bf T$ for a weighted ensemble simulation, using lag times longer than a single resampling cycle can cause the dataset to become biased. Trajectories that are ended prematurely (by merging) are generally closer to stable states. If $\bf T$ simply disregards these trajectories, it is systematically undercounting transitions towards more stable states. We have proved this implicit bias in the MBC-MSM article (link to pre-print XXX) and name this as $\bf merging bias$.   

*Solution:* Keep an accounting of all "incomplete" trajectory segments as follows.  When constructing a time-lagged count matrix $\bf C$ with a lag-time $\tau$, define a set of matrices, ${\bf M}_i$, whose elements $M_i(k,j)$ count the number of trajectories that have made it to $k$, starting from state $j$, but there are still $i$ timesteps left in the transition interval.

The complete time-lagged count matrix is then:

$\bf C$ = $\bf C_{obs}$ + $\sum$ $\bf T_i$ $\bf M_i$

where, the summation runs over i=1 to i=$\tau$-1, the ${\bf C}_{\text{obs}}$ is the observed set of counts without merging and ${\bf T}_i$ is the $i$-step transition matrix, resulting from the normalizing the full set of counts.

**Technical details - **

*Contents:* Codes that can be used to build merging corrected time-lagged counts matrices, $\bf C$, following the method proposed in the MBC-MSM paper (link to pre-print XXX).

Software prerequisite: WEPY.

*Prior requirements from the user:* 
We assume that the users have already

(i) run the WE simulations or have access to WE simulation data that will be used to build the markov models,

(ii) have identified the important features and clustered the WE data into a number of states as per the MSM methodology. 

*Workflow:*
(i)    We adjust the WE weights in such a way that the effect of merging is taken out i.e., the dumping of killed walker weights into the survivor walker is not allowed. This is called 'merge_adjusted' weights.

(ii)   Use the sliding_windows function from wepy to build a time lagged dataset with a given lag-time. This dataset does not include time-lagged points where merging has taken place in between the points within that 
lag-time interval.

(iii)  Build the c_obs matrix,  1-step transition probability matrix and the M dictionaries from the WE simulation data.
