using ThreadPinning
pinthreads(:cores)
threadinfo()

using BenchmarkTools, NeuralOperators, Random, Optimisers, Zygote

# TODO: Add training code

# FNO
n_points = 128
batch_size = 64

x = rand(Float32, 1, n_points, batch_size);
y = rand(Float32, 1, n_points, batch_size);
data = [(x, y)];
t_fwd = zeros(5)
t_train = zeros(5)

for i in 1:5
    chs = (1, 128, fill(64, i)..., 128, 1)
    model = FourierNeuralOperator(; ch=chs, modes=(16,), σ=gelu)
    # model(x) # TTFX

    # t_fwd[i] = @belapsed $model($x)

    # t_train[i] = @belapsed train!($model, $ps, $st, $data; epochs=10)
end

println("\n## FNO (Flux NeuralOperators.jl)")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end

# DeepONets
eval_points = 128
batch_size = 64
dim_y = 1
m = 32

u = rand(Float32, m, batch_size);
y = rand(Float32, dim_y, eval_points, batch_size);

g = rand(Float32, eval_points, batch_size);

data = [((u, y), g)]
t_fwd = zeros(5)
t_train = zeros(5)
for i in 1:5
    ch_branch = (m, fill(64, i)..., 128)
    ch_trunk = (dim_y, fill(64, i)..., 128)
    model = DeepONet(ch_branch, ch_trunk)
    # model(u, y) # TTFX

    # t_fwd[i] = @belapsed $model($u, $y)

    # t_train[i] = @belapsed train!($model, $ps, $st, $data; epochs=10)
end

println("\n## DeepONet (Flux NeuralOperators.jl)")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end
