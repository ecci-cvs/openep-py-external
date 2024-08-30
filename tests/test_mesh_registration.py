import sys
import os
import vedo
import pycpd
import time
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import openep

case_files_path = '/home/lab/Documents/openep-reg-test-datasets/'
source_case_file_name = '103.mat'
target_file_name1 = '119.mat'

source_case = openep.load_openep_mat(f"{case_files_path}{source_case_file_name}")
target_case = openep.load_openep_mat(f"{case_files_path}{target_file_name1}")
source_case.points /= 100
target_case.points /= 100

source_case.cpd_registration_init('rigid', target_case)

plt = vedo.Plotter(size=(800, 800), axes=0, bg='lightblue', title="")

iter_text = vedo.Text2D("Iteration: 0", pos="top-right", c='black')
iter_text.name = "iter_text"

source_mesh = vedo.Mesh(source_case.create_mesh()).c('r5')
source_mesh.name = "source_mesh"    

target_mesh = vedo.Mesh(target_case.create_mesh()).c('b7')
target_mesh.name = "target_mesh" 

global iter
iter = 0
def run_single_iteration(w=None,e=None, plt=plt, iter_text=iter_text, source_case=source_case):
    global iter
    iter += 1
    print("run_single_iteration")
    start_time = time.time() 
    source_case.cpd_registration_run_single_iteration()
    end_time = time.time()

    plt.remove('source_mesh')
    source_mesh = vedo.Mesh(source_case.create_mesh()).c('r5')
    source_mesh.name = "source_mesh"      
    plt += source_mesh	
    plt.remove('iter_text')
    duration = end_time - start_time
    iter_text = vedo.Text2D(f"Iteration: {iter}, duration: {duration:.2f} s", pos="top-right", c='black')
    iter_text.name = "iter_text"
    # print(f"**********Iteration: {iter}**********")
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
          ],
        #   camera=camera,
        ).close()