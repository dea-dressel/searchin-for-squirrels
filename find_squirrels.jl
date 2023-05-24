using GaussianProcesses, CSV, DataFrames, GeoStats


data = CSV.read("nyc_squirrels.csv", DataFrame)

lat_min = minimum(data[:,"lat"])
lat_max = maximum(data[:,"lat"])
long_min = minimum(data[:,"long"])
long_max = maximum(data[:,"long"])

lower_left = LatLon(lat_min, long_min)
lower_right = LatLon(lat_min, long_max)
upper_left = LatLon(lat_max, long_min)

x_range = haversine(lower_left, lower_right)
y_range = haversine(lower_left, upper_left)

println("lower_left: ", lower_left)
println("lower_right: ", lower_right)
println("upper_left: ", upper_left)
println("x_range: ", x_range)
println("y_range: ", y_range)




# println("min lat: ", minimum(data[:,"lat"]))
# println("max lat: ", maximum(data[:,"lat"]))
# println("lat diff: ", )
# println("min long: ", minimum(data[:,"long"]))
# println("max long: ", maximum(data[:,"long"]))

# # observed x and y values
# x_obs = []
# y_obs = []

# for row in data
#     push!( (row["lat"],row["long"]), x_obs )


# end