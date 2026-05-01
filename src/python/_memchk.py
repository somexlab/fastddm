# SPDX-FileCopyrightText: 2023-present University of Vienna
# SPDX-License-Identifier: GPL-3.0-or-later

import psutil


def get_free_mem() -> int:
    """Return the available memory.

    Returns
    -------
    int
        Available memory on RAM.
    """
    return int(psutil.virtual_memory().available)
