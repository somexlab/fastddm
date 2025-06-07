# Copyright (c) 2023-2025 University of Vienna.
# Part of FastDDM, released under the GNU GPL-3.0 License.

# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

import subprocess as sp
from typing import List


def get_free_gpu_mem() -> List[int]:
    """Ruturn available GPU memory.

    Returns
    -------
    List[int]
        Available GPU memory for each device.
    """
    # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow/59571639#59571639

    # retrieve free gpu memory
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]

    # convert memory from MB to bytes and create list
    memory_free_values = [1048576 * int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    return memory_free_values
