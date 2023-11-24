import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from tqdm import tqdm


class TestbedDataset(InMemoryDataset):
    def __init__(self, root=None, dataset=None,
                 pro=None, poc=None,y=None, transform=None,
                 pre_transform=None,smile_graph=None):


        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(pro, poc,y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass


    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):

        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self, pro, poc, y,smile_graph):

        data_list = []

        # for name in smile_graph:
        for name in tqdm(smile_graph, desc="Processing"):
            protein = pro[name]
            pocket = poc[name]
            labels = y[name]

            c_size, features, edge_index = smile_graph[name]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(0,1),
                                y=torch.FloatTensor([labels]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.protein = torch.LongTensor([protein])
            GCNData.pocket = torch.LongTensor([pocket])

            GCNData.pdbid = name
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        # print(data_list)
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

