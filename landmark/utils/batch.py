"""Class to create batch of data"""
import pandas as pd

class Batch(object):

    def __init__(self, batch_size, data, label=None, nb_epoch=1):
        self.data = data
        self.label = label
        self.nb_data = data.shape[0]
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.current_min = 0
        self.current_max = self.batch_size
        self.batch_data = self.data[:batch_size]
        if self.label is not None:
            self.batch_label = self.label[:batch_size]

    @property
    def next(self):
        self.current_min += self.batch_size
        self.current_max += self.batch_size
        if self.current_max > self.nb_data:
            if self.label is None:
                self.end(has_label=False)
            else:
                self.end(has_label=True)
        else:
            self.batch_data = self.data[self.current_min:self.current_max]

    def end(self, has_label):
        if self.nb_epoch != 1:
            self.batch_data = self.data[self.current_min:]
            if has_label:
                self.batch_label = self.label[self.current_min:]
            self.current_min = 0
            self.current_max = self.current_max - self.nb_data
            self.batch_data = pd.concat([self.batch_data, self.data[:self.current_max]],
                                        axis=1)
            if has_label:
                self.batch_label = pd.concat([self.batch_label, self.label[:self.current_max]],
                                             axis=1)

        else:
            self.batch_data = self.data[self.current_min:]
            if has_label:
                self.batch_label = self.label[self.current_min:]
            self.current_max = 0
            self.current_min = -self.batch_size
