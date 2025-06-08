# SPDX-FileCopyrightText: 2023-present University of Vienna
# SPDX-FileCopyrightText: 2023-present Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino
# SPDX-License-Identifier: GPL-3.0-or-later

# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

import psutil


def get_free_mem() -> int:
    """Return the available memory.

    Returns
    -------
    int
        Available memory on RAM.
    """
    return psutil.virtual_memory().available
