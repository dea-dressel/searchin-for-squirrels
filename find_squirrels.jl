using GaussianProcesses, Plots, CSV, DataFrames, Geodesy #, PlotlyJS


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

println("origin_ENU: ", origin_ENU)
println("lower_right_ENU: ", lower_right_ENU)
println("upper_left_ENU: ", upper_left_ENU)

x_range = ceil(lower_right_ENU[1])
y_range = ceil(upper_left_ENU[2])

println("x_range: ", x_range)
println("y_range: ", y_range)

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

println(sum(grid))

println("length(x_axis): ", length(x_axis))
println("length(y_axis): ", length(y_axis))
# println(grid[70:100, 70:100])

plt = contourf(x_axis, y_axis, grid', levels =20, c = cgrad(:viridis, rev = true), legend = true, title = "contour", xlabel = "x_bins", ylabel = "y_bins")
display(plt)
savefig("contour_100")
