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

import pathlib
import openep


data_directory = pathlib.Path('/home/ps21/github/cvi42_to_mesh/cvi42_to_mesh/_datasets/')
cvi_workspace = data_directory / 'cvi42wsx_workspaces' / 'IHD001.cvi42wsx'
dicoms_directory = data_directory / 'dicoms' / 'IHD001'
epi_mesh, endo_mesh, dicoms = openep.load_circle_cvi(cvi_workspace, dicoms_directory, return_dicoms_data=True)
