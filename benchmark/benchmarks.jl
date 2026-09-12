using NeuralOperators, BenchmarkTools
using Lux, Random, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# =============================================================================
# FourierNeuralOperator
# =============================================================================

fno1 = FourierNeuralOperator(; chs = (2, 8, 8, 8, 4), modes = (8,), shift = false)
fno2 = FourierNeuralOperator(; chs = (2, 8, 8, 8, 4), modes = (4, 4), shift = true)

ps1, st1 = Lux.setup(rng, fno1)
ps2, st2 = Lux.setup(rng, fno2)

x1 = rand(rng, Float32, 32, 2, 4)
x2 = rand(rng, Float32, 16, 16, 2, 4)

SUITE["fno"] = BenchmarkGroup()

SUITE["fno"]["construct_1d"] = @benchmarkable FourierNeuralOperator(
    ; chs = (2, 8, 8, 8, 4), modes = (8,)
)
SUITE["fno"]["setup_1d"] = @benchmarkable Lux.setup($rng, $fno1)
SUITE["fno"]["forward_1d"] = @benchmarkable $fno1($x1, $ps1, $st1)
SUITE["fno"]["forward_2d"] = @benchmarkable $fno2($x2, $ps2, $st2)

# =============================================================================
# DeepONet
# =============================================================================

don = DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16))
psd, std = Lux.setup(rng, don)

xin = (rand(rng, Float32, 64, 5), rand(rng, Float32, 1, 10))

SUITE["deeponet"] = BenchmarkGroup()

SUITE["deeponet"]["construct"] = @benchmarkable DeepONet(
    ; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16)
)
SUITE["deeponet"]["forward"] = @benchmarkable $don($xin, $psd, $std)
