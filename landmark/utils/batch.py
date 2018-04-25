"""Class to create batch of data"""
import pandas as pd

class Batch(object):

    def __init__(self, batch_size, data, batch_start=0, label=None, nb_epoch=1):
        self.data = data
        self.label = label
        self.nb_data = data.shape[0]
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.batch_start = batch_start
        self.current_min = self.batch_start*self.batch_size
        self.current_max = (self.batch_start + 1)*self.batch_size
        self._create

    @property
    def next(self):
        self.current_min += self.batch_size
        self.current_max += self.batch_size
        self._create

    @property
    def _create(self):
        if self.current_max > self.nb_data:
            if self.label is None:
                self._end(has_label=False)
            else:
                self._end(has_label=True)
        else:
            self.batch_data = self.data[self.current_min:self.current_max]
            if self.label is not None:
                self.batch_label = self.label[self.current_min:self.current_max]

    def _end(self, has_label):
        if self.nb_epoch != 1:
            self.batch_data = self.data[self.current_min:]
            if has_label:
                self.batch_label = self.label[self.current_min:]
            self.current_max = self.current_max - self.nb_data
            self.batch_data = pd.concat([self.batch_data, self.data[:self.current_max]],
                                        axis=0)
            if has_label:
                self.batch_label = pd.concat([self.batch_label, self.label[:self.current_max]],
                                             axis=0)
            self.current_min = self.current_max - self.batch_size

        else:
            self.batch_data = self.data[self.current_min:]
            if has_label:
                self.batch_label = self.label[self.current_min:]
            # Case where you should have only 1 epoch
            # pointers are update to re-start from the beginning if it continues
            self.current_max = 0
            self.current_min = -self.batch_size
