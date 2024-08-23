# Fourier Neural Operators (FNOs)

FNOs are a subclass of Neural Operators that learn the learn the kernel $\Kappa_{\theta}$, parameterized on $\theta$ between function spaces:

```math
(\Kappa_{\theta}u)(x) = \int_D \kappa_{\theta}(a(x), a(y), x, y) dy  \quad \forall x \in D
```

The kernel makes up a block $v_t(x)$ which passes the information to the next block as:
```math
v^{(t+1)}(x) = \sigma((W^{(t)}v^{(t)} + \Kappa^{(t)}v^{(t)})(x))
```

FNOs choose a specific kernel $\kappa(x,y) = \kappa(x-y)$, converting the kernel into a convolution operation, which can be efficiently computed in the fourier domain.

```math
\begin{align*}
(\Kappa_{\theta}u)(x) 
&= \int_D \kappa_{\theta}(x - y) dy  \quad \forall x \in D\\
&= \mathcal{F}^{-1}(\mathcal{F}(\kappa_{\theta}) \mathcal{F}(u))(x) \quad \forall x \in D
\end{align*}
```
where $\mathcal{F}$ denotes the fourier transform. Usually, not all the modes in the frequency domain are used with the higher modes often being truncated.
## Usage


## API
```@docs
FourierNeuralOperator
```