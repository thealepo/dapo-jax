import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax

from reward import RewardModel  # NOTE: change to a simple reward function (non-transformer-based)

BETA = 0.01
MAX_NEW_TOKENS = 8  # placeholder
MU = 4
G = 8

def generate_groups(policy , input_ids , prompt_len , rng):
    def gen_one(rng , _):
        rng , rng_gen = jax.random.split(rng)
        output = policy.generate(input_ids , rng=rng_gen , max_new_tokens=MAX_NEW_TOKENS)
        return rng , outputs

    _ , outputs = jax.lax.scan(gen_one , rng , None , length=G)
    outputs = rearrange(outputs , 'g b t -> b g t')
    responses = outputs[: , : , prompt_len:]
    return outputs , responses

def compute_advantages(rewards):
    mean = rewards.mean(axis=1 , keepdims=True)
    std = rewards.std(axis=1 , keepdims=True)
    return (rewards - mean) / std

def compute_log_probs(policy , outputs , prompt_len):
    flattened = rearrange(outputs , 'b g t -> (b g) t')
    log_probs = policy.log_probs_of(flattened)
    log_probs = rearrange(log_probs , '(b g) t -> b g t' , g=G)
    return log_probs[: , : , prompt_len:]

def dapo_loss(log_probs_rl , old_log_probs , advantages , epsilon_low=0.2 , epsilon_high=0.28):
    ratio = jnp.exp(log_probs_rl - old_log_probs)  # [batch , G , response_len]
    At = rearrange(advantages , 'b g -> b g 1')

    clipped = jnp.clip(ratio , 1-epsilon_low , 1+epsilon_high)
    surrogate = jnp.minimum(ratio * At , clipped * At)

    loss = -surrogate.mean()
    return loss

@nnx.jit(static_argnames=('prompt_len',))
def train_step(policy , optimizer , outputs , old_log_probs , log_probs_sft , advantages , prompt_len):
    def loss_fn(policy):
        log_probs_rl = compute_log_probs(policy , outputs , prompt_len)
        return dapo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages)

    loss_val , grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(policy , grads)
    return loss_val


# generate groups -> collect rewards -> dynamic sample check (DAPO) -> calculate advantages -> loss
def train_batch(policy , reward , optimizer , input_ids , prompt_len , rng):
    rng , rng_gen = jax.random.split(rng)

    full_generations , responses = generate_groups(policy , input_ids , prompt_len , rng_gen)  # [batch , G , total_len] | [batch , G , response_len]

    old_log_probs = compute_log_probs(policy , full_generations , prompt_len)
    
    flat_responses = rearrange(responses , 'b g t -> (b g) t')
    flat_mask = jnp.ones_like(flat_responses)  # change the masking logic
    flat_rewards = reward(flat_responses , flat_mask)

    rewards = rearrange(flat_rewards , '(b g) -> b g' , g=G)

    # NOTE: Dynamic Sampling
    reward_sum = rewards.sum(axis=1)
    keep = (reward_sum > 0) & (reward_sum < G)  # [batch]

    full_generations = full_generations[keep]
    old_log_probs = old_log_probs[keep]
    rewards = rewards[keep]

    advantages = compute_advantages(rewards)

    losses = []
    for _ in range(MU):
        loss = train_step(policy , optimizer , full_generations , old_log_probs , advantages , prompt_len)
        losses.append(loss)

    return losses












    
