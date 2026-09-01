# DAPO Implementation in Flax NNX
Flax NNX implementation of Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) (what an acronym)

## DAPO

Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) is a post-training reinforcement learning algorithm designed to target several limitations of GRPO. The objective of DAPO is defined as:

<PUT IN LATER>

## The Changes

The core changes that differentiate DAPO from GRPO are outlined in the following.

### Epsilon Clipping Bound

DAPO introduces a higher upper epsilon bound. 

### Dynamic Sampling

Perhaps the biggest code-related change (and to me, the most fun!) is the inclusion of Dynamic Sampling. This change targets the quality of the samples from our rollouts. Consider an RLVR example, in which a response containing the correct answer is scored with a 1, and a 0 otherwise. If in the G groups, a

<CONTINUE WITH THE EXPLANATIONS>
