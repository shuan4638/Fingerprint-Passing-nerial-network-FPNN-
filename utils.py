from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


def get_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #mol = AllChem.AddHs(mol)
    bi = {}   
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=1024, bitInfo=bi)
    return mol, bi

def fp2string(bi, max_len):
    biy  = {}
    for i in bi:
        for j in bi[i]:
            if j[1] == 1 and j[0] not in biy:
                biy[j[0]] = i
    arr = np.zeros(max_len)
    for b in biy:
        arr[b] = biy[b]
    return arr

def get_neighbor(mol, min_len, max_len):
    neighbors = {}
    mol_len = 0
    for atom in mol.GetAtoms():
        neighbors[atom.GetIdx()] = []
        mol_len += 1
    for bond in mol.GetBonds():
        atom1_node = bond.GetBeginAtom().GetIdx()
        atom2_node = bond.GetEndAtom().GetIdx()
        neighbors[atom1_node].append(atom2_node)
        neighbors[atom2_node].append(atom1_node)
    A = np.zeros((max_len, max_len))
    if mol_len < min_len or mol_len > max_len:
        return 0, A, A

    def norm_A(A_):
        d = np.diag(np.power(np.array(A_.sum(1)), -0.5).flatten(), 0)
        a_norm = d.dot(A_).dot(d)
        return a_norm    
    
    def padding(A, max_len):
        result = np.zeros((max_len,max_len))
        result[:A.shape[0],:A.shape[1]] = A
        return result
    
    A_ = np.zeros((mol_len, mol_len))
    for i in neighbors:
        A_[i][i] = 1
        A_[i][neighbors[i]] = 1
        A[i][neighbors[i]] = 1       
    a_norm = padding(norm_A(A_), max_len) 
    return mol_len, A, a_norm

def smiles2grpah(smiles, min_len, max_len):
    mol, info = get_info(smiles)
    mol_len, A, A_ = get_neighbor(mol, min_len, max_len)
    if mol_len == 0:
        return 0, 'n', 'n', 'n'
    arr = fp2string(info, max_len)
    return mol_len, A, arr, A_

def load_data(file, min_len = 5, max_len = 40):
    data_type = file.split('.')[-1]
    if data_type == 'csv':
        df = pd.read_csv(file, header = None)
    else:
        df = pd.read_csv(file, delimiter = '\t', header = None)
    data = pd.DataFrame(index = ['X','A', 'A_', 'y'])
    for i in df.index:
        smiles = df[0][i]
        y = df[1][i]
        try:
            mol_len, A, arr, A_ = smiles2grpah(smiles, min_len, max_len)
        except:
            print (smiles, 'cannot be converted')
            mol_len = 0
        if mol_len != 0:
            data[i] = arr, A, A_, y
        else:
            pass
    data_T = data.T
    data_X = np.array([i for i in data_T['X']])
    data_A = np.array([i for i in data_T['A']])
    data_A_ = np.array([i for i in data_T['A_']])
    data_y = np.array([i for i in data_T['y']])
    return data_X, data_A, data_A_, data_y