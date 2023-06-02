
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
const PLOT = true
const DEBUG = false
const CONSTRAINT = true
const CONSTRAINT_TYPE = "eating" #options include: "eating", "running", "climbing" 
const KERNEL_TYPE = "SE" #"Matern" 

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

    ## GAUSSIAN PROCESS FITTING ##

    # define hyperparameters
    length_scale = 100
    ll = [log(length_scale), log(length_scale)]
    data_noise = 0.2
    logObsNoise = log(data_noise)
    m = MeanZero()

    # define kernel
    if KERNEL_TYPE == "SE"
        kernel = SE(ll, logObsNoise)
    elseif KERNEL_TYPE == "Matern"
        kernel = Matern(1/2, ll, logObsNoise)
    else
        throw("ERROR: Unknown kernel type")
    end

    # initialize our observed x float64 matrix
    xy_obs = init_xy_obs(x_axis, y_axis)
    
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

    ### PREDICT ###

    # extract predicted mean and confidence interval
    μ, σ² = predict_y(gp, xy_all)
    σ = sqrt.(σ²)
    confid95 = 1.96*σ
    

    if DEBUG
        println("size(xy_all): ", size(xy_all))
        println("size(μ): ", size(μ))
        println("size(σ²): ", size(σ²))
    end

    ## GAUSSIAN PROCESS OPTIMIZATION ##
    xy_best = xy_all[:,findmax(μ)[2]]

    # NEEDS FIXING
    #optimal_xy = LLAfromENU(upper_left_LLA, origin_LLA, wgs84)
    

    println("\n~~ OPTIMIZATION OUTPUT ~~")
    println("Parameters: ")
    println("   Bin Size: $(bin_size)")
    println("   Characteristic Length Scales: $(ll)")
    println("   Measurement Noise: $(data_noise)")
    println("   Kernel: $(KERNEL_TYPE)")
    print("Constrained: $(CONSTRAINT)")
    if CONSTRAINT
        print(" , with constraint $(CONSTRAINT_TYPE)")
    end
    println("\nBest Location: $(xy_best)")
    
    
    if PLOT
        create_and_save_plots(x_axis, y_axis, z_grid, x_pred, y_pred, xy_all, μ, xy_best)
    end



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
            if row[CONSTRAINT_TYPE] #constrainted to constraint type
                # increment count of squirrels within that bin
                z_grid[floor(Int64, x)+1, floor(Int64, y)+1] += 1
            end
        else #no constraint 
            # increment count of squirrels within that bin
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


"""
    Arguments:
        - `x_axis`: (Vector) x values from 0 to x_range with bin_size increments
        - `y_axis`: (Vector) y values from 0 to y_range with bin_size increments
        - `z_grid`: (Matrix) squirrel counts in 2D binned grid space
        - `x_pred`: (Vector) Randomized x prediction points
        - `y_pred`: (Vector) Randomized y prediction points
        - `xy_all`: (Vector) Randomized x,y prediction points
        - `μ`:      (Vector) Predicted means
"""
function create_and_save_plots(x_axis, y_axis, z_grid, x_pred, y_pred, xy_all, μ, xy_best)
    # Create plot of observed data
    plt = contourf(x_axis, y_axis, z_grid', 
                    levels = 20, 
                    legend = true,
                    c = cgrad(:davos, rev=true),  
                    title = "\nObserved Squirrel Counts with Eating Constraint\n", xlabel = "x", ylabel = "y", 
                    label="Observed Squirrel Counts", 
                    titlefontsize=8,
                    guidefontsize=8,
                    tickfontsize=8,
                    legendfontsize=8,
                    aspectratio=:equal,
                    xlim=(0,2700), ylim=(0,4000),
                    dpi=100)

    savefig("final_plots/observed_squirrel_counts_constrained=$(CONSTRAINT)")

    # Overlay predicted mean contour lines
    μ_grid = reshape(μ, (length(y_pred), length(x_pred)))

    contour!(x_pred, y_pred, μ_grid, 
            levels=20, 
            c=cgrad(:sunset, rev=true), 
            legend=true, 
            title="Observed Squirrel Counts \n& GP Predicted Mean Contours", xlabel="x", ylabel="y",
            label="GP Predicted Mean",
            aspectratio=:equal,
            titlefontsize=8,
            guidefontsize=8,
            tickfontsize=8,
            legendfontsize=8,
            xlim=(0,2700), ylim=(0,4000),
            dpi=100)

    savefig("final_plots/observed_and_predicted_mean_kernel=$(KERNEL_TYPE)_constrained=$(CONSTRAINT)")

    # Overlay predicted mean contour lines
    scatter!([xy_best[x_index]], [xy_best[y_index]],
            title="Observed Squirrel Counts, \nGP Predicted Mean Contours & Optimal Location\n", xlabel="x", ylabel="y",
            label="Optimal Location",
            legend=:best,
            markershape=:star5,
            aspectratio=:equal,
            titlefontsize=8,
            guidefontsize=8,
            tickfontsize=8,
            legendfontsize=5,
            xlim=(0,2700), ylim=(0,4000),
            dpi=100)

    savefig("final_plots/observed_and_predicted_mean_and_optimal_spot_kernel=$(KERNEL_TYPE)_constrained=$(CONSTRAINT)")

    # Create predicted mean contour plot
    plt = contourf(x_pred, y_pred, μ_grid, 
                    levels=20, 
                    c=cgrad(:sunset, rev=true), 
                    title="\nGP Predicted Mean\n", xlabel="x", ylabel="y",
                    titlefontsize=8,
                    guidefontsize=8,
                    tickfontsize=8,
                    legendfontsize=5,
                    aspectratio=:equal,
                    xlim=(0,2700), ylim=(0,4000),
                    dpi=100)

    savefig("final_plots/predicted_mean_kernel=$(KERNEL_TYPE)_constrained=$(CONSTRAINT)")

    # Create predict mean surface plot
    plt = surface(xy_all[1,:], xy_all[2,:], μ, 
                    title = "\nGP Predicted Mean w/ $(KERNEL_TYPE) Kernel\n", xlabel = "x", ylabel = "y", zlabel="Predicted Mean Squirrels",
                    c=cgrad(:sunset, rev=true),
                    titlefontsize=11,
                    guidefontsize=8,
                    tickfontsize=8,
                    legendfontsize=8,
                    xlim=(0,2700), ylim=(0,4000),
                    dpi=100)

    savefig("final_plots/predicted_mean_surface_kernel=$(KERNEL_TYPE)_constrained=$(CONSTRAINT)")

end


# call main function
main()





