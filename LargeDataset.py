import numpy as np
import os


'''
Xshape: lista de listas que contienen la forma y dtype del dato
'''

class LargeDataset():
    def __init__(self, dataset_name, Xshape, Yshape):
        super().__init__()

        self.Xshape = Xshape
        self.Yshape = Yshape
        self.dataset_name = dataset_name

        if os.path.exists(os.path.join(dataset_name, 'DS_METADATA.npy')):
            with open(os.path.join(dataset_name, 'DS_METADATA.npy'), 'rb') as file:
                self.parameters2save = np.load(file)
        else:
            self.parameters2save = np.zeros((2), dtype=int)

        self.retX = []
        self.retY = []

        for shape in Xshape:
            self.retX.append(np.zeros(shape[0], dtype=shape[1]))

        for shape in Yshape:
            self.retY.append(np.zeros(shape[0], dtype=shape[1]))

        if not os.path.isdir(dataset_name):
            os.mkdir(dataset_name)

        self.bs_counter = 0

    def append(self, X, Y):
        with open(os.path.join(self.dataset_name, f'SAMPLE_{self.parameters2save[0]}.npy'), 'wb') as file:
            for d in X:
                np.save(file, d)

            for d in Y:
                np.save(file, d)

        self.parameters2save[0] += 1
        if self.parameters2save[0] > self.parameters2save[1]:
            self.parameters2save[1] = np.copy(self.parameters2save[0])

        with open(os.path.join(self.dataset_name, 'DS_METADATA.npy'), 'wb') as file:
            np.save(file, self.parameters2save)

    def __getitem__(self, idx:int):
        with open(os.path.join(self.dataset_name, f'SAMPLE_{idx}.npy'), 'rb') as file:
            for i in range(len(self.Xshape)):
                self.retX[i] = np.load(file)

            for i in range(len(self.Yshape)):
                self.retY[i] = np.load(file)

        return self.retX, self.retY

    def next_batch(self, batch_size):
        print(f'Getting slice of data [{self.bs_counter}, {self.bs_counter + batch_size})')

        retX = []
        retY = []

        if batch_size + self.bs_counter > self.parameters2save[1]:
            batch_size = self.parameters2save[1] - self.bs_counter

        for shape in self.Xshape:
            retX.append(np.zeros((batch_size, *shape[0]), dtype=shape[1]))

        for shape in self.Yshape:
            retY.append(np.zeros((batch_size, *shape[0]), dtype=shape[1]))

        for i in range(self.bs_counter, self.bs_counter + batch_size, 1):
            Xs, Ys = self[i]

            for idsm, d in enumerate(Xs):
                retX[idsm][i-self.bs_counter] = d

            for idsm, d in enumerate(Ys):
                retY[idsm][i-self.bs_counter] = d

        self.bs_counter = self.bs_counter + batch_size

        if self.bs_counter >= self.parameters2save[1]:
            self.bs_counter = 0

        return retX, retY
