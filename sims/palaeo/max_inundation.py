import shutil
from thetis import *
import os.path
import sys
import math
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import params

# where should the output of this analysis go
output_dir = 'analysis'
create_directory(output_dir)

# where is the output of your model?
thetis_dir = params.output_dir

# was this run created with the DumbCheckpoint code? If so, make this True
legacy_run = False

# You *MAY* need to edit below this line
# Make sure below matches your main run file as much as possible
# *if* anything goes wrong with the analysis
#============================================================#

# making an assumption here on what the hdf5 output is called
chk = CheckpointFile("output/hdf5/Elevation2d_00000.h5",'r')
thetis_mesh = chk.load_mesh()

chk = CheckpointFile('bathymetry.h5','r')
bathymetry2d = chk.load_function(thetis_mesh,'bathymetry')
chk.close()
t_start = params.start_time
# How long does your simulations run for (s)
t_end = params.end_time #40 days (i.e. 30 days of analysis)
# how often are exports produced in the main run?
t_export = params.output_time

t_n = int((t_end - t_start) / t_export) + 1
thetis_times = t_start + t_export*np.arange(t_n)

# --- create solver ---
solverObj = solver2d.FlowSolver2d(thetis_mesh, Constant(10))
options = solverObj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory =  thetis_dir
options.manning_drag_coefficient = Constant(1.0)
options.horizontal_viscosity = Constant(1.0)
options.element_family = "dg-dg"

P1CG = FunctionSpace(thetis_mesh, "CG", 1)

# Initialize a CG function for running max elevation with a low baseline value
max_fs_cg = Function(P1CG, name='MaxFS_CG')
max_fs_cg.dat.data[:] = -9999.0 

count = 0
for t in thetis_times:
    PETSc.Sys.Print('Reading h5 files. Time ', int((t-t_start)/t_export), t)
    solverObj.load_state(int((t-t_start)/t_export), legacy_mode=legacy_run)
    
    # 1. Project the instantaneous, smooth DG elevation snapshot to CG space
    elev_cg_t = project(solverObj.fields.elev_2d, P1CG)
    
    # 2. Update the running maximum across CG nodes vectorially (instantaneous)
    max_fs_cg.dat.data[:] = np.maximum(max_fs_cg.dat.data[:], elev_cg_t.dat.data[:])
    
    count += 1

# max_fs_cg is now a perfectly smooth continuous envelope of the maximums.
# You can now cleanly subtract your bathymetry to find maximum inundation depth.
b_cg = project(bathymetry2d, P1CG)
md = Function(P1CG, name='Max_Depth')
md.interpolate(max_fs_cg - (-1 * b_cg))

with CheckpointFile(output_dir + '/max_depth.h5', "w") as chk:
    chk.save_mesh(thetis_mesh)
    VTKFile( output_dir + '/max_depth.pvd').write(md)
    VTKFile( output_dir + '/bathy_test.pvd').write(b_cg)
    chk.save_function(md, name='Max_Depth')

