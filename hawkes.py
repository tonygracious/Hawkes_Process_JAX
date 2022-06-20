# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from  jax import grad, jit, vmap
from jax import random
import numpy as np
import jax
#import jax.scipy.optimize as jsp_opt
from jax_minimize_wrapper import minimize
from DataGenerator import data_generator, batch_events_merge, Event_batching


'''
 \lambda_i = \mu_i +  \sum_j \sum_{t_{jl} < t} \alpha_{ij} \omega \exp{ \omega (t - t_{jl} ) }
'''


# A helper function to randomly initialize to initialize parameters
def random_layer_params(m, key):
    mu_key, alpha_key, omega_key = random.split(key, 3)
    mu = random.uniform(mu_key, (m,))
    alpha = random.uniform(alpha_key, (m, m))
    omega = random.uniform(omega_key)
    return [mu, alpha, omega]

def intensity(params, event_times, event_types, t):
    """
    params: parameters of multivarariate hawkes
    event times
    """
    mu = params[0]
    alpha = params[1]
    omega = params[2]
    lbda = []
    for i in range(m):
        lbda_i = mu[i]
        alpha_i = alpha[i, event_types]
        lbda_i += jnp.sum( alpha_i  * omega *  jnp.exp( -1 * ( omega * (t - event_times) ) ) )
        lbda.append(lbda_i)
    lbda = jnp.stack(lbda)
    return lbda.sum()


def intensity_int(params, event_times, event_types, event_mask, end_time):
    mu = params[0]
    alpha = params[1]
    omega = params[2]
    m = mu.shape[0]
    T = end_time
    lbda_int = 0
    for i in range(m):
        lbda_int += mu[i] * T
        alpha_i = alpha[i, event_types]
        lbda_int -=  jnp.sum( alpha_i * (jnp.exp(-1 * ( omega * (T - event_times))) - 1) * event_mask )
    return lbda_int


def neg_loglikelihood(params, event_times, event_types, event_mask, end_time):
    mu =  params[0]
    alpha =  params[1]
    omega =  params[2]
    N = event_times.shape[0]
    log_lbda = 0
    for n in range(N):
        i = event_types[n]
        lbda_n = mu[i]
        alpha_i = alpha[i, event_types[:n]]
        lbda_n += jnp.sum( alpha_i * omega * jnp.exp(-1 * (omega * (event_times[n] - event_times[:n])))  )
        log_lbda += jnp.log(lbda_n + 1e-7) * event_mask[n]
    lbda_int =  intensity_int(params, event_times, event_types, event_mask, end_time)
    return (log_lbda - lbda_int) * (-1)
@jit
def loss_function(params, event_times, event_types, event_mask, end_time):
    batch_negloglikelihood =vmap(neg_loglikelihood, in_axes=(None, 0, 0, 0, None))
    loss = batch_negloglikelihood(params, event_times, event_types, event_mask, end_time)
    return loss.mean()

if __name__ == '__main__':
    m = 3
    decays = [.3]
    baseline = [0.5, 0.1, 0.5]
    adjacency = [[[0.1], [.0], [0]],
                 [[0], [.3], [0]],
                 [[0], [0], [0.5]]]
    end_time = 100
    key = random.PRNGKey(0)
    #params = random.uniform(key, (m + m**2 + 1,) )
    params = random_layer_params(m, key)
    n_realizations = 5
    N_events = data_generator(baseline, decays, \
                              adjacency, end_time, n_realizations)
    # Batching Events with appropriate masks
    event_merge = batch_events_merge(N_events)
    batch_events, batch_event_type, batch_pad = Event_batching(event_merge)
    event_times = jnp.array(batch_events)
    event_types = jnp.array( batch_event_type, dtype=jnp.int32)
    event_mask  = jnp.array(batch_pad, dtype=jnp.int32)

    print("Before Optimization Negative Loglikelihood", loss_function(params, event_times, event_types, event_mask, end_time ))
    bounds = [(0, 1) for i in range(m+m**2+1)]
    opt = {"maxiter": 50}
    result = minimize(loss_function, params, args = (event_times, event_types, event_mask, end_time), method = "L-BFGS-B", bounds=bounds, options=opt)
    print("After Optimization Negative Loglikelihood", loss_function(result.x, event_times, event_types, event_mask, end_time))
    print("mu True values", np.array(baseline))
    print("mu Estimated Values", result.x[0])
    print("alpha True values", np.array(adjacency).reshape(m, m))
    print("alpha Estimated Values", result.x[1])
    print("omega true value", decays[0])
    print("omega estimate value", result.x[2])

