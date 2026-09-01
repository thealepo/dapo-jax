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

    rewards = rearrange(flat_rewards , '(b g) t -> b g t' , g=G)













    
