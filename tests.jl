using GaussianProcesses, Plots, CSV, DataFrames, Geodesy, LinearAlgebra


m = MeanZero()
se = SE(0.0,0.0)

X = 2π*rand(2, 10)
println("X: ", size(X))
y = sin.(X[1,:]) .* cos.(X[2,:]) + 0.5*rand(10)
println("y: ", size(y))
gp2 = GP(X,y,m,se)
# Plot mean and variance
p1 = plot(gp2; title="Mean of GP")
p2 = plot(gp2; var=true, title="Variance of GP", fill=true)
plot(p1, p2; fmt=:png)

