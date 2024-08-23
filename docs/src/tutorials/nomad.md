# Nonlinear Manifold Decoders for Operator Learning (NOMADs) 
NOMADs are similar to DeepONets in the aspect that they can learn when the input and output function spaces are defined on different domains. Their architecture is different and use nonlinearity to the latent codes to obtain the operator approximation.
The architecture involves an approximator to encode the input function space, which is directly concatenated with the input function coordinates, and passed into a decoder net to give the output function at the given coordinate.

```math
\begin{align*}
u(y) \xrightarrow{\mathcal{A}} & \; \beta \\
& \quad \searrow\\
&\quad \quad \mathcal{G}_{\theta} u(y) = \mathcal{D}(\beta, y) \\
&  \quad \nearrow \\
y 
\end{align*}
```


## Usage

## API
```@docs
NOMAD
```