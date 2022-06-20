# Hawkes_Process_JAX
Hawkes Process MLE Inference Using JAX
* Run python hawkes.py
* Requires [tick](https://x-datainitiative.github.io/tick/) and [JAX](https://jax.readthedocs.io/en/latest/)



*   Hawkes Process model for event modelling https://en.wikipedia.org/wiki/Hawkes_process

```math
\lambda_i (t) = \mu_i +  \sum_j \sum_{t_{jl} < t} \alpha_{ij} \omega \exp{  \omega (t - t_{jl} ) }
```
Loglikelihood is defineds as,

```math
\sum_{n=1}^N {\log(\lambda_{i_{t_n}} (t_n) )} - \sum_i \int_0^T \lambda_i (t) dt 
```
* Before Optimization Negative Loglikelihood 268.909910517387
* After Optimization Negative Loglikelihood 219.8147895512949
* \mu True values ```[0.5 0.1 0.5]```
* \mu Estimated Values ```[0.53409991 0.04881035 0.4799496 ]```
* \alpha True values 
```math
   [[0.1 0.  0. ]
    [0.  0.3 0. ]
    [0.  0.  0.5]]
```
* \alpha Estimated Values 
```math
 [[0.01708269 0.         0.04558697]
 [0.04051707 0.37785955 0.        ]
 [0.07921349 0.2555025  0.53284138]]
```
* \omega true value ```0.3```
* \omega estimate value ```0.22339805387220335```


