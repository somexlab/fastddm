import psutil

def get_free_mem() -> int:
    """Get available memory

    Returns
    -------
    int
        Available memory on RAM.
    """
    return psutil.virtual_memory().available
