# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

import os
import numpy as np

from fastddm.imagestructurefunction import ImageStructureFunction
from fastddm.azimuthalaverage import AzimuthalAverage

this_file_path = os.path.abspath(os.path.dirname(__file__))
contin_exec_path = os.path.join(this_file_path, "contin")

def _run_contin(
        input_file: str,
        output_file: str
        ) -> None:
    """Run CONTIN executable.

    Executes the command:
    contin < input_file > output_file

    Parameters
    ----------
    input_file : str
        Input file to contin executable
    output_file : str
        File to which contin output is redirected
    """
    exec_file = contin_exec_path
    full_comm = exec_file + " < " + input_file + " > " + output_file

    os.system(full_comm)
