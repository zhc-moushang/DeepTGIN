import pandas as pd
from rdkit import Chem
import networkx as nx
from dataset import *



def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(str(atom.GetChiralTag()),['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER','misc'])+
                    one_of_k_encoding_unk(atom.GetFormalCharge(),[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'])+
                    one_of_k_encoding_unk(atom.GetNumRadicalElectrons(),[0, 1, 2, 3, 4, 'misc'])+
                    one_of_k_encoding_unk(str(atom.GetHybridization()),['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'])+
                    [atom.IsInRing()]+
                    [atom.GetIsAromatic()]
    )

def mol2_to_graph(mol2):
    mol = Chem.MolFromMol2File(mol2)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x
def poc_cat(prot):
    x = np.zeros(63)
    for i, ch in enumerate(prot[:63]):
        x[i] = seq_dict[ch]
    return x

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

data_select = 'test2016'

mol2_path = '/pdbbind2020/'+data_select+'/mol2/'
mol2_file_list = os.listdir(mol2_path)
smile_graph = {}
protein_dic = {}
pocket_dic = {}
smiles_dic = {}
error = []
protein_df = pd.read_csv('/sequence/'+data_select+'_protein.csv')
proteins = {i["id"]: i["seq"] for _, i in protein_df.iterrows()}

pocket_df = pd.read_csv('/sequence/'+data_select+'_pocket.csv')
pockets = {i["id"]: i["seq"] for _, i in pocket_df.iterrows()}



for mol2_file in tqdm(mol2_file_list, desc="Processing"):

    try:

        name = mol2_file.split('.')[0]
        g = mol2_to_graph(mol2_path + mol2_file)
        smile_graph[name] = g

        protein_dic[name] = seq_cat(proteins[name])
        pocket_dic[name] = poc_cat(pockets[name])

    except:
        error.append(mol2_file.split('_')[0])
        continue


affinity = {}
affinity_df = pd.read_csv( 'affinity_data.csv')
for _, row in affinity_df.iterrows():
    affinity[row[0]] = row[1]

test_data = TestbedDataset(root='data', dataset=data_select,pro = protein_dic,poc = pocket_dic,y=affinity,smile_graph=smile_graph)




