# DeepONets

DeepONets are another class of networks that learn the mapping between two function spaces by encoding the input function space and the location of the output space. The latent code of the input space is then projected on the location laten code to give the output. This allows the network to learn the mapping between two functions defined on different spaces.


```math
\begin{align*}
u(y) \xrightarrow{\text{branch}} & \; b \\
& \quad \searrow\\
&\quad \quad \mathcal{G}_{\theta} u(y) = \sum_k b_k t_k \\
&  \quad \nearrow \\
y \; \; \xrightarrow{\text{trunk}} \; \; & t  
\end{align*}
```
## Usage

## API
```@docs
DeepONet
```