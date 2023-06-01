
#=
    Searching for Squirrels
    AA 222
    Mary Cooper, Dea Dressel, and Chris Osgood.

    Just three friends trying to find the best squirrel viewing location
    in Central Park, NYC. We fit a 2D Guassian Process to our
    squirrel data, and then implement expected improvement exploration
    to find the optimal locaion for seeing the highest number of squirrels. 
    
=#


using GaussianProcesses, Plots, CSV, DataFrames, Geodesy, LinearAlgebra, Distributions

const data_file = "nyc_squirrels.csv"

# FLAGS FOR PRINTING AND PLOTTING
const PLOT_CONTOUR = true
const DEBUG = true
const CONSTRAINT = false
const constraint_type = "eating" #options include: "eating", "running", "climbing" 

const x_index = 1
const y_index = 2

function main()

    ## DATA MANAGEMENT ##
    
    # read in and understand squirrel data
    data = CSV.read(data_file, DataFrame)
    origin_LLA, origin_ENU, lower_right, upper_left = find_data_origin_and_boundaries(data)

    # hyperparameter
    bin_size = 100

    # gather x and y axis information
    x_range = ceil(lower_right[x_index])
    y_range = ceil(upper_left[y_index])
    

    x_axis, y_axis = get_axes(x_range, y_range, bin_size)

    x_bins = collect(x_axis)
    y_bins = collect(y_axis)

    num_x = length(x_bins) # vector dimensions for x design points
    num_y = length(y_bins) # vector dimensions for y design points

    # build grid to store lat/long squirrel counts (our z value)
    z_grid = build_z_grid(data, origin_LLA, num_x, num_y, bin_size, x_range, y_range)

    # plotting
    if PLOT_CONTOUR
        plt = contourf(x_axis, y_axis, z_grid', levels = 20, c = cgrad(:viridis, rev = true), legend = true, title = "contour", xlabel = "x", ylabel = "y")
        # display(plt)
        savefig("contour_plots/contour_$(bin_size)_constrained=$(CONSTRAINT)")
    end

    ## GAUSSIAN PROCESS FITTING ##

    # define hyperparameters
    length_scale = 100
    ll = [log(length_scale), log(length_scale)]
    data_noise = 0.5
    logObsNoise = log(data_noise)
    m = MeanZero()
    kernel = SE(ll, logObsNoise)
    # kernel = Matern(1/2, ll, logObsNoise)

    # initialize our observed x float64 matrix
    xy_obs = init_xy_obs(x_axis, y_axis)
    println("xy_obs[:,1:5]: ", xy_obs[:,1:5])

    # reshape z_grid into float64 vector to pass as y_obs to GP
    z_obs = reshape(z_grid', (:,))

    if DEBUG
        println("size(z_obs): ", size(z_obs))
        println("z_grid[:,1:5]: ", z_grid[:, 1:5])
        println("z_obs[1:15]: ", z_obs[1:15])
    end

    # fit Gaussian Process
    gp = GP(xy_obs, z_obs, m, kernel, logObsNoise)

    # initialize our prediction design point matrix
    xy_all, x_pred, y_pred = init_xy_all(x_range, y_range, bin_size)

    if DEBUG
        println("xy_all[:,1:5]: ", xy_all[:,1:5])
    end

    # extract predicted mean and confidence interval
    μ, σ² = predict_y(gp, xy_all)
    σ = sqrt.(σ²)
    confid95 = 1.96*σ
    
    μ_grid = reshape(μ, (length(y_pred), length(x_pred)))


    if DEBUG
        println("size(xy_all): ", size(xy_all))
        println("size(μ): ", size(μ))
        println("size(σ²): ", size(σ²))
    end

    # plot GP predict
    if DEBUG
        println("xy_all[:,10:15]: ", xy_all[:, 10:15])
        println("sum(μ): ", sum(μ))
        println("μ[10:15]: ", μ[10:15])
        println("σ[10:15]: ", σ[10:15])
    end
    if PLOT_CONTOUR
        plt = contour!(x_pred, y_pred, μ_grid, levels=20, c=cgrad(:cool, rev=true), legend=true, title="Prediction", xlabel="x", ylabel="y")
        savefig("contour_plots/contour_predictedmean_$(bin_size)_constrained=$(CONSTRAINT)")

        plt = contourf(x_pred, y_pred, μ_grid, levels=20, c=cgrad(:cool, rev=true), legend=true, title="Prediction", xlabel="x", ylabel="y")
        savefig("contour_plots/predictedmean_$(bin_size)_constrained=$(CONSTRAINT)")

        plt = surface(xy_all[1,:], xy_all[2,:], μ, title = "surface", xlabel = "x", ylabel = "y", label="predicted mean")
        # surface!(xy_all[1, :], xy_all[2, :], μ - confid95, fillrange=μ + confid95, fillalpha=0.5,
        #     label="95% confidence interval", legend=true)
        # gui(plt)
        # println("Look at the plot and then type something to move on")
        # readline()
        savefig("surface_plots/predictedmean_$(bin_size)_constrained=$(CONSTRAINT)")
    end


    ## IMPLEMENT EXPLORATION STRATEGY HERE ##

end

"""
    Arguments:
    - `data`: (DataFrame) squirrel data in central park

    Returns:
        - origin of data in LLA
        - origin of data in ENU
        - lower right coordinate of data in ENU
        - upper left coordinate of data in ENU
"""
function find_data_origin_and_boundaries(data)
    
    # find min/max lat/long of data
    lat_min = minimum(data[:,"lat"])
    lat_max = maximum(data[:,"lat"])
    long_min = minimum(data[:,"long"])
    long_max = maximum(data[:,"long"])

    # create LLA descriptions
    # origin_LLA = LLA(lat_max, long_min, 0.0)
    # lower_left_LLA = LLA(lat_min, long_min, 0.0)
    # upper_right_LLA = LLA(lat_max, long_max, 0.0)
    
    # create LLA descriptions
    origin_LLA = LLA(lat_min, long_min, 0.0)
    lower_right_LLA = LLA(lat_min, long_max, 0.0)
    upper_left_LLA = LLA(lat_max, long_min, 0.0)

    # convert LLA to ENU
    origin_ENU = ENU(origin_LLA, origin_LLA, wgs84)
    lower_right_ENU = ENU(lower_right_LLA, origin_LLA, wgs84)
    upper_left_ENU = ENU(upper_left_LLA, origin_LLA, wgs84)

    if DEBUG
        println("\n~~ DATA MANAGEMENT ~~")
        println("origin_ENU: ", origin_ENU)
        println("lower_right_ENU: ", lower_right_ENU)
        println("upper_left_ENU: ", upper_left_ENU)

    end
    
    return origin_LLA, origin_ENU, lower_right_ENU, upper_left_ENU
end


"""
    Arguments:
    - `x_range`: Maximum x_axis value for data
    - `y_range`: Maximum y_axis value for data
    - `bin_size`: Size of grid bins

    Returns:
        - (Vector) x values from 0 to x_range with bin_size increments
        - (Vector) y values from 0 to y_range with bin_size increments
"""
function get_axes(x_range, y_range, bin_size)
    
    # values from 0 to range with bin_size increments
    x_axis = range(0.0, stop=x_range, step=bin_size)
    y_axis = range(0.0, stop=y_range, step=bin_size)

    if DEBUG
        println("\n~~ GRID BUILDING ~~")
        println("x_range: ", x_range)
        println("y_range: ", y_range)
        println("bin_size: ", bin_size)
        println("length(x_axis): ", length(x_axis))
        println("length(y_axis): ", length(y_axis))
    end
    return x_axis, y_axis

end

"""
    Arguments:
        - `data`: DataFrame 
        - `origin_LLA`: Origin of data in LLA
        - `num_x_bins`: Number of bins on x_axis
        - `num_y_bins`: Number of bins on y_axis
        - `bin_size`: Size of each bin

    Returns:
        - (Matrix) squirrel counts in 2D binned grid space
"""
function build_z_grid(data, origin_LLA, num_x_bins, num_y_bins, bin_size, x_range, y_range)
    
    z_grid = zeros(num_x_bins, num_y_bins)

    # loop through each row/each squirrel in data set
    for row in eachrow(data)

        # convert from LLA to xyz
        lla = LLA(row["lat"], row["long"], 0.0)
        enu = ENU(lla, origin_LLA, wgs84)
        
        # find bin corressponding to current squirrel's location
        x,y = enu[x_index]/bin_size, enu[y_index]/bin_size
        x,y = max(x,0), max(y,0) # ensure positive x,y values
        
        if CONSTRAINT
            if row[constraint_type] #constrainted to constraint type
                # increment count of squirrels within that bin
                # z_grid[floor(Int64, y)+1, floor(Int64, x)+1] += 1
                z_grid[floor(Int64, x)+1, floor(Int64, y)+1] += 1
            end
        else #no constraint 
            # increment count of squirrels within that bin
            # z_grid[y_range - floor(Int64, y)+1, x_range - floor(Int64, x)+1] += 1
            z_grid[floor(Int64, x)+1, floor(Int64, y)+1] += 1
        end
    end

    if DEBUG
        println("z_grid.size: ", size(z_grid))
    end
    return z_grid
end

"""
    Arguments:
        - `x_axis`: (Vector) x values from 0 to x_range with bin_size increments
        - `y_axis`: (Vector) y values from 0 to y_range with bin_size increments

    Returns:
        - (Vector) reshaped (x,y) observation point pairs
"""
function init_xy_obs(x_axis, y_axis)
    
    xs = Vector{Float64}(undef, 0)
    ys = Vector{Float64}(undef, 0)

     # make every combination of x,y pairs form xs and ys
    for x in x_axis
        for y in y_axis
            push!(xs, x)
            push!(ys, y)
            # println("\nX: ", x)
            # println("Y: ", y)
        end
    end

    # get into expected Vector structure for GP
    xy_obs = vcat(xs', ys')


    if DEBUG
        println("\n~~ FITTING GP ~~")
        println("size(xy_obs):", size(xy_obs))
        # println("xy_obs: ", xy_obs)
    end
    return xy_obs
end


"""
    Arguments:
        - `x_range`: Maximum x_axis value for data
        - `y_range`: Maximum y_axis value for data
        - `num_x`: Number of desired x prediction values 
        - `num_y`: Number of desired y prediction values 

    Returns:
        - (Vector) Randomized x,y prediction points
"""
function init_xy_all(x_range, y_range, bin_size)

    x_rand = collect(0:0.5*bin_size:x_range)
    y_rand = collect(0:0.5*bin_size:y_range)

    xs_all = Vector{Float64}(undef, 0)
    ys_all = Vector{Float64}(undef, 0)
    
    # make every combination of x,y pairs form rand_x and rand_y 
    for x in x_rand
        for y in y_rand
            push!(xs_all, x)
            push!(ys_all, y)
        end
    end

    # get into expected Vector structure for GP
    xy_all = vcat(xs_all', ys_all')

    if DEBUG
        println("typeof(xy_all): ", typeof(xy_all))
        println("size(xy_all): ", size(xy_all))
    end
    return xy_all, x_rand, y_rand
end


# call main function
main()





