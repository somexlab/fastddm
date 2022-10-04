import numpy as np

class ImageStructureFunction:
    def __init__(self):
        self.isf = np.array([[[]]])
        self.lags = np.array([])
        self.taus = np.array([])
        self.kmap = np.array([[]])
        self.qmap = np.array([[]])
        self.dt = 1.0
        self.pixel_size = 1.0
        self.tau_units = ''
        self.q_units = ''
        self.pixel_units = ''
        self.comments = ''
        self.log = []
    
    