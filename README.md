# DAPO Implementation in Flax NNX
Flax NNX implementation of Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) (what an acronym)

## DAPO

Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) is a post-training reinforcement learning algorithm designed to target several limitations of GRPO. The objective of DAPO is defined as:

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)
=
\mathbb{E}\left[
\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left(
r_{i,t}(\theta)\hat{A}_i,\;
\operatorname{clip}\left(
r_{i,t}(\theta),
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}
\right)\hat{A}_i
\right)
\right]
$$

where

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
},
\qquad
\hat{A}_i
=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)}
$$

and dynamic sampling retains only groups satisfying

$$
0 <
\left|
\left\{
o_i : \operatorname{is\_equivalent}(a,o_i)
\right\}
\right|
< G.
$$
