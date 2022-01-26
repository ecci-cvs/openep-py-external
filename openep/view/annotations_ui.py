# OpenEP
# Copyright (c) 2021 OpenEP Collaborators
#
# This file is part of OpenEP.
#
# OpenEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>


"""
Create a dock widget for the annotation viewer.
"""

from PySide6 import QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.widgets
import matplotlib.pyplot as plt
import numpy as np

import openep.draw.draw_routines
from .custom_widgets import CustomDockWidget, CustomNavigationToolbar

plt.style.use('ggplot')

class AnnotationWidget(CustomDockWidget):
    """A dockable widget for annotating electrograms."""

    def __init__(self, title: str):

        super().__init__(title)
        self._init_main_window()
        self.create_sliders()
        self.initialise_slider_values()
    
    def _init_main_window(self):
        
        self.main  = QtWidgets.QMainWindow()
        
        # The dock is set to have bold font (so the title stands out)
        # But all other widgets should have normal weighted font
        main_font = QtGui.QFont()
        main_font.setBold(False)
        self.main.setFont(main_font)
        
        # The central widget will hold a matplotlib canvas and toolbar.
        # The canvas widget will also contain a QComboBox for selecting
        # the electrogram to annnotate.
        self.canvas, self.figure, self.axes = self._init_canvas()
        self.egm_selection, egm_selection_layout = self._init_selection()
        canvas_layout = QtWidgets.QVBoxLayout(self.canvas)
        canvas_layout.addLayout(egm_selection_layout)
        canvas_layout.addStretch()

        toolbar = CustomNavigationToolbar(
            canvas_=self.canvas,
            parent_=self,
        )

        # Setting nested layouts
        central_widget = self._init_central_widget(self.canvas, toolbar)
        self.main.setCentralWidget(central_widget)
        self.setWidget(self.main)
        
    def _init_canvas(self):
        """
        Create an interactive matploblib canvas.
        """

        figure, axes = plt.subplots(ncols=1, nrows=1)
        figure.set_facecolor("white")

        # hide the axis until we have data to plot
        axes.axis('off')
        # and only display x coordinate in the toolbar when hovering over the axis
        axes.format_coord = lambda x, y: f"{x} ms"

        canvas = FigureCanvas(figure)
        
        return canvas, figure, axes

    def _init_selection(self):
        """Create a layout with widgets for selecting which electrogram to annotate.
        """

        annotate_selection = QtWidgets.QComboBox()
        annotate_selection.setMinimumWidth(220)
        #annotate_selection.setStyleSheet('selection-background-color: red')
        #annotate_selection.setStyleSheet('border: 1px solid #d8dcd6; background-color: white;')

        annotate_selection.setStyleSheet(
            "QWidget{"
            "background-color: white;"
            "selection-background-color: #168CFF;"
            "border: 1px solid #d8dcd6;"
            "}"
        )


        annotate_selection_layout = QtWidgets.QHBoxLayout()
        annotate_selection_layout.addWidget(annotate_selection)
        annotate_selection_layout.addStretch()

        return annotate_selection, annotate_selection_layout

    def _init_central_widget(self, canvas, toolbar):
        """Create a placeholder widget to hold the toolbar and canvas.
        """
        
        central_layout = QtWidgets.QVBoxLayout()
        central_layout.addWidget(canvas)
        central_layout.addWidget(toolbar)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(central_layout)
        central_widget.setStyleSheet("border-width: 0px; border: 0px; background-color: #d8dcd6;")
        
        return central_widget


    def create_sliders(self, valmin=0, valmax=1, valstep=1):
        """Add a window of interest range slider to the canvas"""
        
        # location of the RangeSlider on the figure
        #axes_loc = [0.175, 0.1, 0.65, 0.022]
        axes_loc = [0.1535, 0.05, 0.7185, 0.01]
        slider_axes = self.figure.add_axes(axes_loc)
        self.woi_slider = matplotlib.widgets.RangeSlider(
            ax=slider_axes,
            label="WOI",
            valmin=valmin,
            valmax=valmax,
            valstep=valstep,
            closedmin=True,
            closedmax=True,
            dragging=True,
            facecolor="xkcd:light grey",
        )
    
    def initialise_slider_values(self, start_woi=0, stop_woi=1):
        """Set default values for the sliders and plot the axvlines"""
        
        #self.woi_slider.set_val([start_woi, stop_woi])
        #self.woi_slider.set_val([start_woi, stop_woi])
        self.woi_slider.set_max(stop_woi)
        self.woi_slider.set_min(start_woi)

        self.woi_slider_lower_limit = self.axes.axvline(
            start_woi,
            color='grey',
            linestyle='-',
            linewidth=3,
            alpha=1,
        )
        
        self.woi_slider_upper_limit = self.axes.axvline(
            stop_woi,
            color='grey',
            linestyle='-',
            linewidth=3,
            alpha=1,
        )

    def remove_sliders(self):
        """Remove the sliders form the canvas, if they exist"""    
        try:
            self.woi_slider.ax.remove()
        except AttributeError:
            pass
    
    def update_axvline_positions(self, values):
        """Plot vertical lines designating the window of interest"""
        
        start_woi, stop_woi = values
        self.woi_slider_lower_limit.set_xdata([start_woi, start_woi])
        self.woi_slider_upper_limit.set_xdata([stop_woi, stop_woi])
        self.figure.canvas.draw_idle()

    def initialise_egm_selection(self, selections):
        """Set the selections available in the QComboBox"""
        
        self.egm_selection.clear()
        self.egm_selection.insertItems(0, selections)

    def plot_signals(self, times, signals, labels):
        """Plot electrogram/ecg signals.
        
        Args:
            times (np.ndarray): Time at each point in the signal
            signals (np.npdarray): Signals to plot. 2D array of shape (n_signals, n_time_points)
            labels (np.ndarray): Name of each signal. We be used to label the y-axis.
        """
        
        # First we need to clear the axis, but we want to keep to previous limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        
        self.axes.cla()
        self.axes.set_yticklabels([])
        self.axes.set_ylim(ylim)
        self.axes.set_xlim(xlim)
        
        # we need to horizontally shift the signals so they don't overlap
        y_start = 2
        y_separation = 4
        separations = y_start + np.arange(signals.shape[0]) * y_separation
        
        self.axes.plot(times, signals.T + separations, color="blue")
        self.axes.set_yticks(separations)
        self.axes.set_yticklabels(labels)

        # Remove the border and ticks
        plt.tick_params(axis='both', which='both', length=0)
        for spine in ['left', 'right', 'top']:
            self.axes.spines[spine].set_visible(False)
        self.axes.spines['bottom'].set_alpha(0.4)
        
        self.canvas.draw_idle()

    def activate_axes(self, xmin, xmax):
        """Show the axes"""
        
        self.axes.axis('on')
        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(0, 12)
    
    def deactivate_axes(self):
        """Clear and hide the axes"""
        
        self.axes.cla()
        self.axes.axis('off')
        self.canvas.draw()












    def update_slider_limits(self, valmin=0, valmax=1):
        """Update the min/max values of the slider"""

        self.woi_slider.valmin = valmin
        self.woi_slider.valmax = valmax
        self.woi_slider.ax.set_xlim(valmin, valmax)

    def update_slider_values(
        self,
        start_woi=0,
        stop_woi=1,
    ):
        """Update the slider values.
        
        Move the range slider values.
        Plot vertical lines at these x values.
        """
        
        self.woi_slider.set_max(stop_woi)
        self.update_axvline_positions(start_woi, stop_woi)