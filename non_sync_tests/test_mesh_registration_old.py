import sys
import os
import vedo
import pycpd
import numpy as np

specific_dir = '/home/lab/Documents/openep-py-external/'
sys.path.append(specific_dir)
import openep

file_path = '/home/lab/Documents/openep-reg-test-datasets/'
case_file_name = '103.mat'
case_file_name1 = '119.mat'
case_file_name2 = 'openep_dataset_1.mat'
case_file_name3 = 'openep_dataset_2.mat'

case = openep.load_openep_mat(f"{file_path}{case_file_name}")
case1 = openep.load_openep_mat(f"{file_path}{case_file_name1}")
# case.init_cpd_registration('rigid', case1)
# exit()
# case2 = openep.load_openep_mat(f"{file_path}{case_file_name2}")
# case3 = openep.load_openep_mat(f"{file_path}{case_file_name3}")

target_mesh = vedo.Mesh(case.create_mesh()).c('r5').decimate(fraction=0.1)
target_mesh.name = "target_mesh"
source_mesh = vedo.Mesh(case1.create_mesh()).c('b7').decimate(fraction=0.1)
source_mesh.name = "source_mesh"
case = None
case1 = None
# mesh2 = case2.create_mesh()
# mesh3 = case3.create_mesh()

target_mesh.vertices /= 100
source_mesh.vertices /= 100

# target_points = np.array(target_mesh.vertices.copy(),dtype=np.float32)
# source_points = np.array(source_mesh.vertices.copy(), dtype=np.float32)
target_points = target_mesh.vertices
source_points = source_mesh.vertices

print('taret points shape:', target_points.shape)
print('source points shape:', source_points.shape)

# reg = AffineRegistration(**{'X': target_points, 'Y': source_points})
# reg = pycpd.DeformableRegistration(**{'X': target_points, 'Y': source_points, 'low_rank': True})
reg = pycpd.AffineRegistration(**{'X': target_points, 'Y': source_points})

plt = vedo.Plotter(size=(800, 800), axes=0, bg='lightblue', title="")

iter_text = vedo.Text2D("Iteration: 0", pos="top-right", c='black')
iter_text.name = "iter_text"

def run_single_iteration(w=None,e=None, iter_text=iter_text, plt=plt, reg=reg, source_mesh=source_mesh):
    print("run_single_iteration")
    # task_manager.run_single_iteration()
    # retrieve_button_callback(None,None,plt)
    reg.iterate()
    source_mesh.vertices = reg.TY
    # source_mesh.vertices = np.random.rand(source_mesh.vertices.shape[0], 3)*100

    # plt -= source_mesh
    plt.remove('source_mesh')
    plt += source_mesh	
    plt.remove('iter_text')
    iter_text = vedo.Text2D(f"Iteration: {reg.iteration}", pos="top-right", c='black')
    iter_text.name = "iter_text"
    print(f"**********Iteration: {reg.iteration}**********")
    plt += iter_text

plt += target_mesh
plt += source_mesh
plt += iter_text

plt.add_button(run_single_iteration, pos=(0.25,0.05), states=[">"], bc=["p5"]*2)

# Camera settings
camera = dict(
    pos=(2, 2, 2),  # Camera position
    focalPoint=(0, 0, 0),  # Focal point
    viewup=(0, 0, 1)  # View up vector
)

plt.show([
    # vedo.shapes.Text2D('Elastic registration'),
          ],
        #   camera=camera,
        ).close()