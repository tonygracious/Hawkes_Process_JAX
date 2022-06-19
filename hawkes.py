import jax.numpy as jnp
from  jax import grad, jit, vmap
from jax import random

from DataGenerator import data_generator, batch_events_merge, Event_batching


'''
 \lambda_i = \mu_i +  \sum_j \sum_{t_{jl} < t} \alpha_{ij} \exp{ -\alpha_{ij} * \omega_{ij} (t - t_{jl} ) }
'''
# A helper function to randomly initialize to initialize parameters
def random_layer_params(m, key):
    mu_key, alpha_key, omega_key = random.split(key, 3)
    mu = random.uniform(mu_key, (m,))
    alpha = random.uniform(alpha_key, (m, m))
    omega = random.uniform(omega_key)
    return [mu, alpha, omega]

def intensity(params, event_times, event_types, t):
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


def intensity_int(params, event_times, event_types, end_time):
    mu = params[0]
    alpha = params[1]
    omega = params[2]
    m = mu.shape[0]
    T = end_time
    lbda_int = 0
    for i in range(m):
        lbda_int += mu[i] * T
        alpha_i = alpha[i, event_types]
        lbda_int -= (1/omega) * jnp.sum( jnp.exp(-1 * (alpha_i * omega * (T - event_times))) - 1)
    return lbda_int


def loglikelihood(params, event_times, event_types, end_time):
    mu = params[0]
    alpha = params[1]
    omega = params[2]
    N = event_times.shape[0]
    log_lbda = 0
    for n in range(N):
        i = event_types[n]
        lbda_n = mu[i]
        alpha_i = alpha[i, event_types[:n]]
        lbda_n += jnp.sum( alpha_i * jnp.exp(-1 * (alpha_i * omega * (event_times[n] - event_times[:n]))) )
        log_lbda += jnp.log(lbda_n)
    lbda_int =  intensity_int(params, event_times, event_types, end_time)
    return log_lbda - lbda_int




if __name__ == '__main__':
    m = 3
    decays = [.3]
    baseline = [0.5, 0.1, 0.5]
    adjacency = [[[0.1], [.0], [0]],
                 [[0], [.3], [0]],
                 [[0], [0], [0.5]]]
    end_time = 100
    n_realizations = 1
    N_events = data_generator(baseline, decays, \
                              adjacency, end_time, n_realizations)
    # Batching Events with appropriate masks
    event_merge = batch_events_merge(N_events)
    event_times = jnp.array(event_merge[0][:,0])
    event_types = jnp.array(event_merge[0][:,1], dtype = jnp.int32)

    params = random_layer_params(m, random.PRNGKey(0))

    import timeit

    start_time = timeit.default_timer()
    grad_param = grad(loglikelihood, 0)(params, event_times, event_types, end_time)
    print(grad_param)
    print("Before Jit", timeit.default_timer() - start_time)

    grad_intensity = jit( grad(loglikelihood, 0))
    start_time = timeit.default_timer()
    grad_param = grad_intensity(params, event_times, event_types, end_time)
    print("Compilation Jit",timeit.default_timer() - start_time)

    print(grad_param)

    start_time = timeit.default_timer()
    grad_param = grad_intensity(params, event_times, event_types, end_time)
    t1 = timeit.Timer('grad_intensity(params, event_times, event_types, end_time)', globals=globals())
    print("After Jit", t1.timeit(number=7), "seconds")

    print(grad_param)





