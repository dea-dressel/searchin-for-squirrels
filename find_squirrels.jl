using GaussianProcesses, Plots, CSV, DataFrames, Geodesy, LinearAlgebra

const data_file = "nyc_squirrels.csv"

const CONTOUR_PLOT = true
const DEBUG = true

const x_index = 1
const y_index = 2

function main()

    ## DATA MANAGEMENT ##
    data = CSV.read(data_file, DataFrame)

    origin_LLA, origin_ENU, lower_right, upper_left = find_origin_and_boundaries(data)

    bin_size = 100

    # gather x and y axis information
    x_range = ceil(lower_right[x_index])
    y_range = ceil(upper_left[y_index])

    x_axis, y_axis = get_axes(x_range, y_range, bin_size)

    x_bins = collect(x_axis)
    y_bins = collect(y_axis)

    # build grid to store lat/long squirrel counts (our z value)
    z_grid = build_z_grid(data, origin_LLA, x_bins, y_bins, bin_size)

    # plotting
    if CONTOUR_PLOT
        plt = contourf(x_axis, y_axis, z_grid', levels =20, c = cgrad(:viridis, rev = true), legend = true, title = "contour", xlabel = "x_bins", ylabel = "y_bins")
        display(plt)
        savefig("contour_plots/contour_no")
    end

    ## GAUSSIAN PROCESS FITTING ##

    # define hyperparameters
    length_scale = 1.0
    logObsNoise = log(1.0)
    m = MeanZero()
    kernel = SE(log(length_scale),0.0)

    # initialize our observed x float64 matrix matrix
    x_obs = init_x_obs(x_axis, y_axis)

    # reshape z_grid into float64 vector to pass as y_obs to GP
    y_obs = reshape(z_grid, (:,))
    if DEBUG println("size(y_obs): ", size(y_obs)) end

    # fit Gaussian Process
    gp = GP(x_obs, y_obs, m, kernel, logObsNoise)

    # initialize our random design point matrix
    x_all = init_x_all(x_range, y_range, x_axis, y_axis)

    # extract predicted mean and confidence interval
    μ, σ² = predict_y(gp, x_all)
    σ = sqrt.(σ²)
    std95 = 2*σ 

    if DEBUG
        println("size(μ): ", size(μ))
        println("size(σ²): ", size(σ²))
    end

end

function find_origin_and_boundaries(data)
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

    if DEBUG
        println("origin_ENU: ", origin_ENU)
        println("lower_right_ENU: ", lower_right_ENU)
        println("upper_left_ENU: ", upper_left_ENU)
    end
    
    return origin_LLA, origin_ENU, lower_right_ENU, upper_left_ENU
end

function get_axes(x_range, y_range, bin_size)
    

    x_axis = range(0.0, stop=x_range, step=bin_size)
    y_axis = range(0.0, stop=y_range, step=bin_size)

    if DEBUG
        println("x_range: ", x_range)
        println("y_range: ", y_range)
        println("length(x_axis): ", length(x_axis))
        println("length(y_axis): ", length(y_axis))
    end
    return x_axis, y_axis

end

function build_z_grid(data, origin_LLA, x_bins, y_bins, bin_size)
    z_grid = zeros(length(x_bins), length(y_bins))

    for row in eachrow(data)
        lla = LLA(row["lat"], row["long"], 0.0)
        enu = ENU(lla, origin_LLA, wgs84)
        x,y = enu[1]/bin_size, enu[2]/bin_size

        x = max(x,0)
        y = max(y,0)
        
        z_grid[floor(Int64, x)+1, floor(Int64, y)+1] += 1
    end

    if DEBUG
        println("z_grid.size: ", size(z_grid))
    end
    return z_grid
end

function init_x_obs(x_axis, y_axis)
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

    if DEBUG
        println("size(x_obs): ", size(x_obs))
    end
    return x_obs
end

function init_x_all(x_range, y_range, x_axis, y_axis)
    # we randomly sample x,y values within our z_grid
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

    if DEBUG
        println("size(x_all): ", size(x_all))
    end
    return x_all
end





# call main function
main()





