import logging
import torch
import torch.utils.data as data
import numpy as np
import json
import os
import gzip
import tqdm
import pubchempy as pcp
from torch_geometric.data import Batch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from collections import OrderedDict
import os
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample,LoggingHandler
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sentence_transformers import models, losses,util,datasets,evaluation
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.data import Data
import networkx as nx
from compound_gnn_model import GNNComplete
from ipdb import set_trace
import argparse
import csv


def smiles_to_graph(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol == None:
        print(smiles)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data