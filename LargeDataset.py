import numpy as np
import os


class LargeDataset():
    def __init__(self, dataset_name, Xshape, Yshape):
        super().__init__()

        self.Xshape = Xshape
        self.Yshape = Yshape
        self.dataset_name = dataset_name

        if not os.path.isdir(dataset_name):
            os.mkdir(dataset_name)


