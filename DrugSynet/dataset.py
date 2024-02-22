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

#定义指纹的位数,nbits用于普通指纹，longbits用于长指纹
nbits = 1024
longbits = 16384
# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
# dictionary
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
#fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
#fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
#fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)


#将smiles编码转换成图数据
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

class SynergyEncoderDataset(data.Dataset):
    def __init__(self, drug_1_smiles,drug_2_smiles,Y_Label,context,maxLen=128, device=torch.device("cuda:0"),fp_name='ecfp4'):
        self.maxLen = maxLen
        self.drug_1_smiles = drug_1_smiles
        self.drug_2_smiles = drug_2_smiles
        self.context = context
        self.Y = Y_Label
        self.device = device
        self.fp_name = fp_name
        self.len = len(self.Y)#具体有多少数据
        self.features = json.loads(open("/home/th_xy/DeepDDs-master/data/allfeature.json", 'r').read())
        #features为每个细胞系里的954个基因的表达 size:(1,954)；
        model_name = 'output/simcsesqrt-model'
        self.drug_model = SentenceTransformer(model_name,device=device)
        self.smiles_to_graph = smiles_to_graph
        self.encode_smiles()
    def __len__( self ):
        return self.len

    def encode_smiles(self):
        self.simse = {}
        self.drugfp = {}
        self.graph = {}
        for smile in set(self.drug_1_smiles).union(set(self.drug_2_smiles)):  
            smile_str = smile[:self.maxLen]
            self.simse[smile] = self.drug_model.encode(smile_str)
            self.graph[smile] = self.smiles_to_graph(smile)
            mol = Chem.MolFromSmiles(smile)
            fp = fpFunc_dict[self.fp_name](mol)
            self.drugfp[smile] = np.asarray(fp)

    def __getitem__(self,index):
         drug1_graph = self.graph[self.drug_1_smiles[index]]
         drug2_graph = self.graph[self.drug_2_smiles[index]]
         drug1_bert = self.simse[self.drug_1_smiles[index]]
         drug2_bert = self.simse[self.drug_2_smiles[index]]
         synergyScore = self.Y[index]
         fp1 = self.drugfp[str(self.drug_1_smiles[index])]
         fp2 = self.drugfp[str(self.drug_2_smiles[index])]
         context_features=self.features[self.context[index]]
         drug1_smile = self.drug_1_smiles[index]
         drug2_smile = self.drug_2_smiles[index]
         cell = self.context[index]

         
         return drug1_smile,drug2_smile,cell,drug1_graph,drug2_graph,[torch.FloatTensor(drug1_bert),torch.FloatTensor(drug2_bert),torch.FloatTensor(fp1),torch.FloatTensor(fp2),torch.FloatTensor([context_features]),torch.LongTensor([synergyScore])]