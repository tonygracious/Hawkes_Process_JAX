import jax.numpy as jnp
from  jax import grad, jit, vmap
from jax import random
import jax

from DataGenerator import data_generator, batch_events_merge, Event_batching


'''
 \lambda_i = \mu_i +  \sum_j \sum_{t_{jl} < t} \alpha_{ij} \exp{ -\alpha_{ij} * \omega_{ij} (t - t_{jl} ) }
'''
# A helper function to randomly initialize to initialize parameters
def random_layer_params(m, key):
    """
    m: num of event types
    key: random generator key for jax
    """
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
    m = mu.shape[0]

    lbda = []
    for i in range(m):
        lbda_i = mu[i]
        alpha_i = alpha[i, event_types]
        lbda_i += jnp.sum( alpha_i  *  jnp.exp( -1 * ( alpha_i * omega * (t - event_times) ) ) )
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
        lbda_int -= (1/omega) * jnp.sum( (jnp.exp(-1 * (alpha_i * omega * (T - event_times))) - 1) * event_mask )
    return lbda_int


def loglikelihood(params, event_times, event_types, event_mask, end_time):
    mu = params[0]
    alpha = params[1]
    omega = params[2]
    N = event_times.shape[0]
    log_lbda = 0
    for n in range(N):
        i = event_types[n]
        lbda_n = mu[i]
        alpha_i = alpha[i, event_types[:n]]
        lbda_n += jnp.sum( alpha_i * jnp.exp(-1 * (alpha_i * omega * (event_times[n] - event_times[:n])))  )
        log_lbda += jnp.log(lbda_n) * event_mask[n]
    lbda_int =  intensity_int(params, event_times, event_types, event_mask, end_time)
    return log_lbda - lbda_int

'''
def block_until_ready(pytree):
  return tree_util.tree_map(lambda x: x.block_until_ready(), pytree)
'''

if __name__ == '__main__':
    m = 3
    decays = [.3]
    baseline = [0.5, 0.1, 0.5]
    adjacency = [[[0.1], [.0], [0]],
                 [[0], [.3], [0]],
                 [[0], [0], [0.5]]]
    end_time = 100
    '''
    n_realizations = 1
    N_events = data_generator(baseline, decays, \
                              adjacency, end_time, n_realizations)
    # Batching Events with appropriate masks
    event_merge = batch_events_merge(N_events)
    event_times = jnp.array(event_merge[0][:,0])
    event_types = jnp.array(event_merge[0][:,1], dtype = jnp.int32)
    '''
    params = random_layer_params(m, random.PRNGKey(0))
    '''
    print('dfadskjf;kaj')
    import timeit

    start_time = timeit.default_timer()
    t1 = timeit.Timer('jax.block_until_ready(grad(loglikelihood, 0)(params, event_times, event_types, end_time))',
                      globals=globals())
    print("Before Jit",  t1.timeit(number=7), "seconds" )
    grad_param  = grad(loglikelihood, 0)(params, event_times, event_types, end_time)
    print(grad_param)

    grad_intensity = jit( grad(loglikelihood, 0))
    start_time = timeit.default_timer()
    grad_param = jax.block_until_ready(grad_intensity(params, event_times, event_types, end_time))
    print("Compilation Jit",timeit.default_timer() - start_time)
    print(grad_param)

    start_time = timeit.default_timer()
    grad_param = grad_intensity(params, event_times, event_types, end_time)
    t1 = timeit.Timer('jax.block_until_ready(grad_intensity(params, event_times, event_types, end_time))', globals=globals())
    print("After Jit", t1.timeit(number=7), "seconds")
    print(grad_param)
    '''
    n_realizations = 5
    N_events = data_generator(baseline, decays, \
                              adjacency, end_time, n_realizations)
    # Batching Events with appropriate masks
    event_merge = batch_events_merge(N_events)
    batch_events, batch_event_type, batch_pad = Event_batching(event_merge)
    event_times = jnp.array(batch_events)
    event_types = jnp.array( batch_event_type, dtype=jnp.int32)
    event_mask  = jnp.array(batch_pad, dtype=jnp.int32)

    #grad_param = grad(loglikelihood, 0)(params, event_times[0], event_types[0], event_mask[0], end_time)

    loglikelihood_compiled  =  jit(vmap(loglikelihood, in_axes =( None, 0, 0, 0, None)))
    loglikelihood_compiled(params, event_times, event_types, event_mask, end_time)
    grad_compiled  = jit(vmap(grad(loglikelihood, 0) , in_axes =( None, 0, 0, 0, None)))
    grad_compiled(params, event_times, event_types, event_mask, end_time)




