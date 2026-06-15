import scipy.interpolate
import numpy as np

fluidity_times = np.loadtxt('../../data/times.txt')
boundary_coords = np.loadtxt('../../data/coordinates.txt')
boundary_data = np.loadtxt('../../data/elev.txt')

def set_tsunami_field(boundary_elev, t):
    
    # Set elev field using nearest neighbour interpolation from boundary points extracted from Fluidity tsunami simulations
    mesh2d = boundary_elev.function_space().mesh()
    xvector = mesh2d.coordinates.dat.data
    evector = boundary_elev.dat.data


    t_storegga = t
    # Set elev field using nearest neighbour interpolation from boundary points extracted from Fluidity tsunami simulations
    fluidity_dt = fluidity_times[1]-fluidity_times[0]
    fluidity_index = int(t_storegga/fluidity_dt)
    boundary_data_1 = boundary_data[fluidity_index]
    boundary_data_2 = boundary_data[fluidity_index+1]
    interpolator_1 = scipy.interpolate.NearestNDInterpolator(boundary_coords, boundary_data_1)
    interpolator_2 = scipy.interpolate.NearestNDInterpolator(boundary_coords, boundary_data_2)
    for i,xy in enumerate(xvector):
        val1 = interpolator_1((xy[0], xy[1]))
        val2 = interpolator_2((xy[0], xy[1]))
        t1 = fluidity_index * fluidity_dt
        t2 = (fluidity_index+1) * fluidity_dt
        val_at_t = (val1 * (t2 - t_storegga) + val2 * (t_storegga - t1)) / fluidity_dt
        evector[i] = val_at_t

    return
