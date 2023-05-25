using GaussianProcesses, Plots, CSV, DataFrames, Geodesy, LinearAlgebra

make_contour = false

data = CSV.read("nyc_squirrels.csv", DataFrame)

lat_min = minimum(data[:,"lat"])
lat_max = maximum(data[:,"lat"])
long_min = minimum(data[:,"long"])
long_max = maximum(data[:,"long"])

origin_LLA = LLA(lat_min, long_min, 0.0)
lower_right_LLA = LLA(lat_min, long_max, 0.0)
upper_left_LLA = LLA(lat_max, long_min, 0.0)

origin_ENU = ENU(origin_LLA, origin_LLA, wgs84)
lower_right_ENU = ENU(lower_right_LLA, origin_LLA, wgs84)
upper_left_ENU = ENU(upper_left_LLA, origin_LLA, wgs84)

# println("origin_ENU: ", origin_ENU)
# println("lower_right_ENU: ", lower_right_ENU)
# println("upper_left_ENU: ", upper_left_ENU)

x_range = ceil(lower_right_ENU[1])
y_range = ceil(upper_left_ENU[2])

# println("x_range: ", x_range)
# println("y_range: ", y_range)

# hyperparameter
bin_size = 100

x_axis = range(0.0, stop=x_range, step=bin_size)
y_axis = range(0.0, stop=y_range, step=bin_size)

x_bins = collect(x_axis)
y_bins = collect(y_axis)

grid = zeros(length(x_bins), length(y_bins))
println("grid.size: ", size(grid))

for row in eachrow(data)
    lla = LLA(row["lat"], row["long"], 0.0)
    enu = ENU(lla, origin_LLA, wgs84)
    x,y = enu[1]/bin_size, enu[2]/bin_size

    x = max(x,0)
    y = max(y,0)
    
    grid[floor(Int64, x)+1, floor(Int64, y)+1] += 1
end

# plotting
if make_contour
    plt = contourf(x_axis, y_axis, grid', levels =20, c = cgrad(:viridis, rev = true), legend = true, title = "contour", xlabel = "x_bins", ylabel = "y_bins")
    display(plt)
    savefig("contour_100")
end

length_scale = 1.0
logObsNoise = log(1.0)
m = MeanZero()
kernel = SE(log(length_scale),0.0)

# initialize our x_obs matrix
xs = Vector{Float64}(undef, 0)
ys = Vector{Float64}(undef, 0)
for x in x_axis
    for y in y_axis
        push!(xs, x)
        push!(ys, y)
    end
end

vals = append!(xs, ys)
x_obs = reshape(vals, (2,:)) 
println("size(x_obs): ", size(x_obs))

# get our grid, squirrel count values into correct shape, call it y_obs
y_obs = reshape(grid, (:,))
println("size(y_obs): ", size(y_obs))

# fit Gaussian Process
gp = GP(x_obs, y_obs, m, kernel, logObsNoise)

# initialize our x_all matrix
# we randomly sample x,y values within our grid
x_rand = rand([0,x_range], length(x_axis))
y_rand = rand([0,y_range], length(y_axis))

xs_all = Vector{Float64}(undef, 0)
ys_all = Vector{Float64}(undef, 0)
for x in x_rand
    for y in y_rand
        push!(xs_all,x)
        push!(ys_all,y)
    end
end
x_all = append!(xs_all,ys_all)
x_all = reshape(x_all, (2,:))
println("size(x_all): ", size(x_all))

# extract predicted mean and confidence interval
μ, σ² = predict_y(gp, x_all)
σ = sqrt.(σ²)
std95 = 2*σ 

println("size(μ): ", size(μ))
println("size(σ²): ", size(σ²))







