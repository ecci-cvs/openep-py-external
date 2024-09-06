import sys
import os
import vedo
import pycpd
import time
import numpy as np
import pickle
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFrame, QLineEdit, QLabel, QFormLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import openep

case_files_path = '/home/lab/Documents/openep-reg-test-datasets/'
source_case_file_name = '103.mat'
target_file_name1 = '119.mat'
MAX_ITERATIONS = 60

source_case = openep.load_openep_mat(f"{case_files_path}{source_case_file_name}")
target_case = openep.load_openep_mat(f"{case_files_path}{target_file_name1}")
source_case.points /= 100
target_case.points /= 100

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        source_case.cpd_registration_init('rigid', target_case)

        self.setWindowTitle("OpenEP Coherent Point Drift Registration Test")
        self.setGeometry(100, 100, 800, 800)

        self.frame = QFrame()
        self.layout = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.plt = vedo.Plotter(qt_widget=self.vtkWidget, size=(800, 800), axes=0, bg='lightblue', title="OpenEP Coherent Point Drift Registration Test")

        iter_text = vedo.Text2D("Iteration: 0", pos="top-right", c='black')
        iter_text.name = "iter_text"

        global source_mesh
        source_mesh = vedo.Mesh(source_case.create_mesh()).c('r5')
        source_mesh.name = "source_mesh"    

        target_mesh = vedo.Mesh(target_case.create_mesh()).c('b7')
        target_mesh.name = "target_mesh" 

        global all_iterations_points
        all_iterations_points = [source_mesh.vertices.copy()]
        global iter
        iter = 0

        self.plt += target_mesh
        self.plt += source_mesh
        self.plt += iter_text

        self.plt.add_button(self.save_movie_button_cb, pos=(0.90,0.95), states=(['Save Movie']), bc=['p5']*2)
        self.plt.add_button(self.save_points_button_cb, pos=(0.55,0.95), states=(['Save']), bc=['p5']*2)
        self.plt.add_button(self.load_points_button_cb, pos=(0.70,0.95), states=(['Load']), bc=['p5']*2)
        self.plt.add_button(self.run_single_iteration_button_cb, pos=(0.2,0.10), states=['Run single iteration'], bc=['p5']*2)
        self.plt.add_button(self.run_all_iterations_button_cb, pos=(0.2,0.05),   states=['Run all iterations  '], bc=['p5']*2)
        self.plt.add_button(self.interpolate_button_cb, pos=(0.55,0.05),   states=['Interpolate'], bc=['p5']*2)
        self.cpd_set_method_button : vedo.Button = self.plt.add_button(self.cpd_set_method_button_cb, pos=(0.8, 0.05), states=['Rigid', 'Affine', 'Deformable'], bc=['g5','g5'])
        self.i_slider : vedo.Slider2D = self.plt.add_slider(self.i_slider_cb, 0, len(all_iterations_points)-1, value=0, pos='top-left', title='View Iteration')
        self.start_i_slider : vedo.Slider2D  = self.plt.add_slider(self.start_i_slider_cb, 0, len(all_iterations_points)-1, value=0, pos=[(0.02,0.3),(0.4,0.3)], title='Start Iteration')
        self.run_i_slider : vedo.Slider2D = self.plt.add_slider(self.run_i_slider_cb, 0, MAX_ITERATIONS, value=0, pos=[(0.02,0.2),(0.4,0.2)], title='Run for Iterations')

        # Camera settings
        camera = dict(
            pos=(2, 2, 2),  # Camera position
            focalPoint=(0, 0, 0),  # Focal point
            viewup=(0, 0, 1)  # View up vector
        )

        # self.vtkWidget.SetRenderWindow(self.plt.window)
        # self.plt.interactor = self.vtkWidget

        # self.plt = vedo.Plotter(offscreen=True)
        self.plt.show()
        # self.plt.show([],
        #         #   camera=camera,
        #         )#.close()
        # self.save_filename_line_edit = QLineEdit()
        # save_filename_line_edit_label = QLabel("Save Filename:")
        # Create a form layout for the line edit and label
        form_layout = QFormLayout()
        self.save_filename_line_edit = QLineEdit('all_iterations_points_deformable.pkl')
        self.load_filename_line_edit = QLineEdit('all_iterations_points_deformable.pkl')
        form_layout.addRow(QLabel("Save File Name:"), self.save_filename_line_edit)
        form_layout.addRow(QLabel("Load File Name:"), self.load_filename_line_edit)
        self.layout.addLayout(form_layout)
        self.layout.addWidget(self.vtkWidget)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show() 
    
    def run_single_iteration_button_cb(self, w=None,e=None):
        global iter
        global all_iterations_points
        iter += 1
        # print(f"run_single_iteration {}")
        start_time = time.time() 
        points = source_case.cpd_registration_run_single_iteration()
        end_time = time.time()

        if iter > len(all_iterations_points) - 1:
            all_iterations_points.append(points.copy())
        else:
            all_iterations_points[iter] = points.copy()

        self.plt.remove('source_mesh')
        source_mesh = vedo.Mesh(source_case.create_mesh()).c('r5')
        source_mesh.name = "source_mesh"      
        self.plt += source_mesh	
        self.plt.remove('iter_text')
        duration = end_time - start_time
        iter_text = vedo.Text2D(f"Last Iteration: {iter}, duration: {duration:.2f} s", pos="top-right", c='black')
        iter_text.name = "iter_text"
        print(f"**********Iteration: {iter}**********")
        self.plt += iter_text
        # return points

    def visualize_cb(self, iteration, error, X, Y):
        print(f"visualize_cb: iteration: {iteration}, error: {error}")
        return False

    def run_all_iterations_button_cb(self, w=None,e=None):
        global iter
        global all_iterations_points
        # all_iterations_points = []
        print("run_all_iterations")
        # source_case.cpd_registration_run_all_iterations(visualize_cb) #TODO

        start_time = time.time() 
        
        iter = int(self.start_i_slider.value)
        i_start = int(self.start_i_slider.value)
        i_end = int(self.start_i_slider.value) + int(self.run_i_slider.value)

        source_case.points = all_iterations_points[i_start].copy()
        source_case.cpd_registration_init(self.cpd_set_method_button.status().lower(), target_case)

        for i in range(i_start, i_end):
            self.run_single_iteration_button_cb()

            curr_time = time.time()
            elapsed_time = curr_time - start_time
            iter_text = vedo.Text2D(f"{'Done. ' if i == i_end-1 else ''}Elapsed Time: {elapsed_time:.2f} s", pos=(0.7, 0.1), c='black')
            iter_text.name = "elapsed_time_text"
            print(f"Elapsed Time: {elapsed_time:.2f} s")
            self.plt.remove('elapsed_time_text')
            self.plt += iter_text
            self.plt.render()
        
        all_iterations_points = all_iterations_points[:i_end+1]
        self.i_slider.range = [0,(len(all_iterations_points)-1)]
        self.start_i_slider.range = [0,(len(all_iterations_points)-1)]    
        print("*****Done*****")    
        # source_mesh = vedo.Mesh(source_case.create_mesh()).c('r5')
        # source_mesh.name = "source_mesh"      
        # plt += source_mesh

    def interpolate_button_cb(self, w=None,e=None):
        print("interpolate_button_cb")

    def update_source_mesh(self):
        global iter
        global source_mesh
        # source_mesh.vertices = np.random.rand(*source_case.points.shape)
        source_mesh.vertices = all_iterations_points[iter].copy()
        self.plt.remove('source_mesh')
        self.plt += source_mesh        

    def i_slider_cb(self, w=None,e=None):
        global iter
        iter = int(w.value)
        # print(f"t_slider_cb {iter}")
        self.update_source_mesh()

    def start_i_slider_cb(self, w=None,e=None):
        # print("start_i_slider_cb")
        pass

    def run_i_slider_cb(self, w=None,e=None):
        # print("run_i_slider_cb")
        pass

    def cpd_set_method_button_cb(self, w=None,e=None):
        # source_case.cpd_registration_init('rigid', target_case)
        w.switch()
        method_name : str = w.status()
        print(f"rigid_button_cb {method_name}")
        source_case.cpd_registration_init(method_name.lower(), target_case)

    def save_points_button_cb(self, w=None,e=None):
        global all_iterations_points
        print("save_points_button_cb")
        with open(self.save_filename_line_edit.text(), 'wb') as f:
            pickle.dump(all_iterations_points, f)

    # def get_slider_by_title(self, title):
    #     for slider in self.plt.sliders:
    #         if slider.title == title:
    #             return slider
    #     return None

    def load_points_button_cb(self, w=None,e=None):
        print("load_points_button_cb")
        global all_iterations_points
        with open(self.load_filename_line_edit.text(), 'rb') as f:
            all_iterations_points = pickle.load(f)
            print(f"Loaded file : {self.load_filename_line_edit.text()} {len(all_iterations_points)} iterations")
            self.i_slider.range = [0,(len(all_iterations_points)-1)]
            self.start_i_slider.range = [0,(len(all_iterations_points)-1)]
            self.update_source_mesh()

    def save_movie_button_cb(self, w=None,e=None):
        print("save_movie_button_cb")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())