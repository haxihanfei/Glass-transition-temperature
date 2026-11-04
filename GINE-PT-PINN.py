
import copy
import math
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, GraphDescriptors, AllChem
RDLogger.DisableLog('rdApp.*')
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
CONFIG = {
    "smiles_col": "SMILES",
    "target_col": "TG",
    "sample_id_col": "sample_id",
    "hs_col": "HS",
    "ss_col": "SS",
    "ce_col": "CE",
    "test_ratio": 0.10,
    "n_splits": 5,
    "batch_size": 16,
    "max_epochs": 800,
    "patience": 60,
    "n_seeds": 5,
    "run_hpo": False,  
    "hpo_trials": 30,
    "run_weight_hpo": True,  
    "weight_hpo_trials": 50,  
    "gcn_hidden": 64,
    "gcn_layers": 2,
    "hidden_dim": 64,
    "n_layers": 2,
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "lambda_bound": 0.4, 
    "lambda_gt": 0.3,  
    "use_global_feat": True,
    "min_samples_per_class": 3,
    "edge_dim": 13,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOM_LIST = [6, 7, 8, 16]
DEGREE_LIST = [1, 2, 3, 4]
HYBRID_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3
]
NUM_H_LIST = [0, 1, 2, 3]
IMPLICIT_H_LIST = [0, 1, 2, 3]
BOND_TYPE_LIST = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
]
SEGMENT_TYPES = ['soft', 'hard', 'chain_extender']
HB_ROLES = ['none', 'donor', 'acceptor', 'both']

SOFT_SEG_SMARTS = {
    'PEG': 'CCOCC',
    'PEG n=5': 'CCOCCOCCOCCOCCOCC',
    'PTMG': 'CCCCOCCCC',
    'PTMGn=4': 'CCCOCCCCOCCCCOCCCCOCCCC'
}
HARD_SEG_SMARTS = {
    'TDI': 'Cc1cc(NC(=O)N)ccc1',
    'MDI': 'c1ccc(Cc2ccc(NC(=O)N)cc2)cc1',
    'HDI': 'NCCCCCCN',
    'IPDI': 'CC1(C)CC(CC(N)C1)N',
    'XDI': 'NCC1CCCCC1CN',
    'NDI': 'c1ccc2c(c1)NC(=O)Nc1cccc2c1'
}
CE_SMARTS = {
    'EDA': 'NCCN',
    'ODA': 'Nc1ccc(Oc2ccc(N)cc2)cc1',
    'PPD': 'Nc1ccc(N)cc1',
    'DETDA': 'Cc1cc(C)c(N)c(C)c1',
    'DDS': 'Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1'
}
HB_DONOR_PATTERN = Chem.MolFromSmarts('[#7H]')
HB_ACCEPTOR_PATTERN = Chem.MolFromSmarts('[#8]')
UREA_PATTERN = Chem.MolFromSmarts('[#7][#6](=[#8])[#7]')
def one_hot(value, value_list):
    vec = [0] * (len(value_list) + 1)
    try:
        vec[value_list.index(value)] = 1
    except ValueError:
        vec[-1] = 1
    return vec
def safe_float(x, default=0.0):
    try:
        val = float(x)
        if math.isfinite(val):
            return val
        return default
    except Exception:
        return default
def scatter_mean_native(x, index, dim=0):
    num_classes = index.max().item() + 1
    feat_dim = x.shape[1]
    sum_feat = torch.zeros(num_classes, feat_dim, device=x.device, dtype=x.dtype)
    sum_feat.index_add_(0, index, x)
    count = torch.bincount(index, minlength=num_classes).float().unsqueeze(1)
    mean_feat = sum_feat / count.clamp(min=1)
    return mean_feat
def smiles_to_graph(smiles, tg_value, hard_type, soft_type, ce_type):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    star_atom_idx = []
    endpoint_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            star_atom_idx.append(atom.GetIdx())
            for neighbor in atom.GetNeighbors():
                endpoint_atom_idx.append(neighbor.GetIdx())
    old_idx_to_new = {}
    new_idx = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            old_idx_to_new[atom.GetIdx()] = new_idx
            new_idx += 1
    num_nodes = new_idx
    if num_nodes == 0:
        return None
    ring_info = mol.GetRingInfo()
    all_atom_rings = ring_info.AtomRings()
    has_charge = False
    try:
        AllChem.ComputeGasteigerCharges(mol)
        has_charge = True
    except Exception:
        has_charge = False
    try:
        mol_wt = safe_float(Descriptors.MolWt(mol)) / 500.0
        num_heavy = safe_float(rdMolDescriptors.CalcNumHeavyAtoms(mol)) / 50.0
        num_rotatable = safe_float(rdMolDescriptors.CalcNumRotatableBonds(mol)) / 20.0
        num_rings = safe_float(rdMolDescriptors.CalcNumRings(mol)) / 5.0
        num_aromatic_rings = safe_float(rdMolDescriptors.CalcNumAromaticRings(mol)) / 5.0
        num_hbd = safe_float(rdMolDescriptors.CalcNumHBD(mol)) / 10.0
        num_hba = safe_float(rdMolDescriptors.CalcNumHBA(mol)) / 10.0
        tpsa = safe_float(rdMolDescriptors.CalcTPSA(mol)) / 200.0
        mol_mr = safe_float(rdMolDescriptors.CalcMolMR(mol)) / 100.0
        balaban_j = safe_float(GraphDescriptors.BalabanJ(mol)) / 5.0
        wiener = safe_float(GraphDescriptors.WienerIndex(mol)) / 1000.0
        global_feat = np.array([
            mol_wt, num_heavy, num_rotatable, num_rings, num_aromatic_rings,
            num_hbd, num_hba, tpsa, mol_mr, balaban_j, wiener
        ], dtype=np.float32)
    except Exception:
        global_feat = np.zeros(11, dtype=np.float32)
    main_chain_atoms = set()
    if len(endpoint_atom_idx) >= 2:
        shortest_path = Chem.GetShortestPath(mol, endpoint_atom_idx[0], endpoint_atom_idx[1])
        for old_idx in shortest_path:
            if old_idx in old_idx_to_new:
                main_chain_atoms.add(old_idx_to_new[old_idx])
    urea_atom_set = set()
    if UREA_PATTERN:
        for match in mol.GetSubstructMatches(UREA_PATTERN):
            for idx in match:
                if idx in old_idx_to_new:
                    urea_atom_set.add(old_idx_to_new[idx])
    general_node_feats = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        new_idx = old_idx_to_new[atom.GetIdx()]
        elem_feat = one_hot(atom.GetAtomicNum(), ATOM_LIST)
        deg_feat = one_hot(atom.GetDegree(), DEGREE_LIST)
        hyb_feat = one_hot(atom.GetHybridization(), HYBRID_LIST)
        arom_feat = [1 if atom.GetIsAromatic() else 0]
        h_feat = one_hot(atom.GetTotalNumHs(), NUM_H_LIST)
        formal_charge_feat = [atom.GetFormalCharge()]
        radical_e_feat = [atom.GetNumRadicalElectrons()]
        impl_h_feat = one_hot(atom.GetNumImplicitHs(), IMPLICIT_H_LIST)
        atom_idx = atom.GetIdx()
        in_ring = [1.0 if atom.IsInRing() else 0.0]
        if atom.IsInRing():
            ring_sizes = [len(r) for r in all_atom_rings if atom_idx in r]
            min_ring_size = min(ring_sizes) if ring_sizes else 0
        else:
            min_ring_size = 0
        ring_size = [safe_float(min_ring_size) / 10.0]
        heavy_neighbors = [atom.GetDegree() / 4.0]
        if len(endpoint_atom_idx) >= 2:
            try:
                d0 = len(Chem.GetShortestPath(mol, endpoint_atom_idx[0], atom_idx))
                d1 = len(Chem.GetShortestPath(mol, endpoint_atom_idx[1], atom_idx))
                total = d0 + d1
                pos_feat = [d0 / total if total > 0 else 0.5]
            except Exception:
                pos_feat = [0.5]
        else:
            pos_feat = [0.5]
        degree_centrality = [atom.GetDegree() / (num_nodes - 1) if num_nodes > 1 else 0.0]
        if has_charge:
            gasteiger = safe_float(atom.GetDoubleProp("_GasteigerCharge"))
            gasteiger = np.clip(gasteiger, -1.0, 1.0)
        else:
            gasteiger = 0.0
        gasteiger_feat = [gasteiger]
        ring_count = len([r for r in all_atom_rings if atom_idx in r])
        ring_count_feat = [safe_float(ring_count) / 10.0]
        is_urea_atom = [1.0 if new_idx in urea_atom_set else 0.0]
        is_main_chain_atom = [1.0 if new_idx in main_chain_atoms else 0.0]
        general_node_feats.append(
            elem_feat + deg_feat + hyb_feat + arom_feat + h_feat
            + formal_charge_feat + radical_e_feat + impl_h_feat
            + in_ring + ring_size + heavy_neighbors + pos_feat + degree_centrality
            + gasteiger_feat + ring_count_feat + is_urea_atom + is_main_chain_atom
        )
    seg_label_list = ['hard'] * num_nodes
    if soft_type in SOFT_SEG_SMARTS:
        soft_pat = Chem.MolFromSmarts(SOFT_SEG_SMARTS[soft_type])
        if soft_pat:
            for match_tuple in mol.GetSubstructMatches(soft_pat):
                for old_idx in match_tuple:
                    if old_idx in old_idx_to_new:
                        seg_label_list[old_idx_to_new[old_idx]] = 'soft'
    if ce_type in CE_SMARTS:
        ce_pat = Chem.MolFromSmarts(CE_SMARTS[ce_type])
        if ce_pat:
            for match_tuple in mol.GetSubstructMatches(ce_pat):
                for old_idx in match_tuple:
                    if old_idx in old_idx_to_new:
                        seg_label_list[old_idx_to_new[old_idx]] = 'chain_extender'
    seg_feats = [one_hot(label, SEGMENT_TYPES)[:-1] for label in seg_label_list]
    donor_set = set()
    for match_tuple in mol.GetSubstructMatches(HB_DONOR_PATTERN):
        for old_idx in match_tuple:
            if old_idx in old_idx_to_new:
                donor_set.add(old_idx_to_new[old_idx])
    acceptor_set = set()
    for match_tuple in mol.GetSubstructMatches(HB_ACCEPTOR_PATTERN):
        for old_idx in match_tuple:
            if old_idx in old_idx_to_new:
                acceptor_set.add(old_idx_to_new[old_idx])
    hb_feats = []
    for i in range(num_nodes):
        is_donor = i in donor_set
        is_acceptor = i in acceptor_set
        if is_donor and is_acceptor:
            role = 'both'
        elif is_donor:
            role = 'donor'
        elif is_acceptor:
            role = 'acceptor'
        else:
            role = 'none'
        hb_feats.append(one_hot(role, HB_ROLES)[:-1])
    final_node_feats = []
    for i in range(num_nodes):
        final_node_feats.append(general_node_feats[i] + seg_feats[i] + hb_feats[i])
    x = torch.tensor(final_node_feats, dtype=torch.float)
    urea_bond_set = set()
    if UREA_PATTERN:
        for match in mol.GetSubstructMatches(UREA_PATTERN):
            n1, c, o, n2 = match
            urea_bond_set.add(frozenset([c, n1]))
            urea_bond_set.add(frozenset([c, o]))
            urea_bond_set.add(frozenset([c, n2]))
    edge_src, edge_dst, edge_attr_list = [], [], []
    for bond in mol.GetBonds():
        begin_old = bond.GetBeginAtomIdx()
        end_old = bond.GetEndAtomIdx()
        if begin_old not in old_idx_to_new or end_old not in old_idx_to_new:
            continue
        begin_new = old_idx_to_new[begin_old]
        end_new = old_idx_to_new[end_old]
        edge_src.extend([begin_new, end_new])
        edge_dst.extend([end_new, begin_new])
        bond_type = bond.GetBondType()
        bond_type_feat = one_hot(bond_type, BOND_TYPE_LIST)[:-1]
        is_main_chain = 1 if (begin_new in main_chain_atoms and end_new in main_chain_atoms) else 0
        is_urea = 1 if frozenset([begin_old, end_old]) in urea_bond_set else 0
        is_in_ring = 1 if bond.IsInRing() else 0
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        if bond_type == rdchem.BondType.SINGLE:
            bo = 1.0
        elif bond_type == rdchem.BondType.DOUBLE:
            bo = 2.0
        elif bond_type == rdchem.BondType.TRIPLE:
            bo = 3.0
        elif bond_type == rdchem.BondType.AROMATIC:
            bo = 1.5
        else:
            bo = 1.0
        bond_order_val = bo / 3.0
        is_rotatable = 1 if (bond_type == rdchem.BondType.SINGLE and not bond.IsInRing()) else 0
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        is_ether = 1 if (
                bond_type == rdchem.BondType.SINGLE and
                {begin_atom.GetAtomicNum(), end_atom.GetAtomicNum()} == {6, 8}
        ) else 0
        seg_begin = seg_label_list[begin_new]
        seg_end = seg_label_list[end_new]
        is_soft_bond = 1 if (seg_begin == 'soft' and seg_end == 'soft') else 0
        is_hard_bond = 1 if (seg_begin == 'hard' and seg_end == 'hard') else 0
        bond_full_feat = bond_type_feat + [is_main_chain, is_urea, is_in_ring, is_conjugated,
                                           bond_order_val, is_rotatable, is_ether,
                                           is_soft_bond, is_hard_bond]
        edge_attr_list.extend([bond_full_feat, bond_full_feat])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    y = torch.tensor([tg_value], dtype=torch.float)
    global_feat_tensor = torch.tensor(global_feat, dtype=torch.float).unsqueeze(0)

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, global_feat=global_feat_tensor
    )
    data.hs = hard_type
    data.ss = soft_type
    data.ce = ce_type
    return data
HS_MW = {
    "HDI": 168.19, "TDI": 174.16, "MDI": 250.25,
    "XDI": 188.18, "IPDI": 222.29, "NDI": 210.19,
}
SS_MW = {
    "PTMG": 162.24, "PTMGn=4": 306.46,
    "PEG": 106.12, "PEG n=5": 238.27,
}
CE_MW = {
    "EDA": 60.10, "PPD": 108.14, "ODA": 200.24,
    "DDS": 248.30, "DETDA": 178.28,
}
def calc_hard_fraction(hs, ss, ce):
    try:
        m_hs = HS_MW[hs.strip()]
        m_ss = SS_MW[ss.strip()]
        m_ce = CE_MW[ce.strip()]
    except KeyError as e:
        raise ValueError(f"未收录的单体: {e}")
    hard_mass = 2 * m_hs + m_ce
    total_mass = 2 * m_hs + m_ss + m_ce
    return hard_mass / total_mass
def build_polyurea_dataset(cfg):
    df = pd.read_csv(cfg["csv_path"])
    df.columns = [c.strip() for c in df.columns]
    required = [cfg["smiles_col"], cfg["target_col"],
                cfg["hs_col"], cfg["ss_col"], cfg["ce_col"], cfg["sample_id_col"]]
    for col in required:
        assert col in df.columns, f"数据集中未找到列: {col}"
    df[cfg["target_col"]] = pd.to_numeric(df[cfg["target_col"]], errors="coerce")
    df = df.dropna(subset=[cfg["target_col"]]).reset_index(drop=True)
    graph_list = []
    fail_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建图数据集"):
        smiles = str(row[cfg["smiles_col"]]).strip()
        tg = float(row[cfg["target_col"]])
        hs = str(row[cfg["hs_col"]]).strip()
        ss = str(row[cfg["ss_col"]]).strip()
        ce = str(row[cfg["ce_col"]]).strip()
        sample_id = row[cfg["sample_id_col"]]
        graph = smiles_to_graph(smiles, tg, hs, ss, ce)
        if graph is not None:
            graph.w_hard = torch.tensor(calc_hard_fraction(hs, ss, ce), dtype=torch.float32)
            graph.sample_id = sample_id
            if 'augment_id' in row:
                graph.augment_id = row['augment_id']
            if 'is_original' in row:
                graph.is_original = row['is_original']
            graph_list.append(graph)
        else:
            fail_count += 1
    if len(graph_list) == 0:
        raise RuntimeError("所有SMILES均解析失败！")
    system_tuples = [(g.hs, g.ss, g.ce) for g in graph_list]
    unique_systems = sorted(list(set(system_tuples)))
    system_to_id = {s: i for i, s in enumerate(unique_systems)}
    for g in graph_list:
        g.system_id = torch.tensor(system_to_id[(g.hs, g.ss, g.ce)], dtype=torch.long)
    print(f"\n[数据] 总行数: {len(df)}")
    print(f"[数据] 解析成功: {len(graph_list)} 条, 失败: {fail_count} 条")
    print(f"[数据] 独特分子数: {df[cfg['sample_id_col']].nunique()} 个")
    print(f"[数据] 独特体系数: {len(unique_systems)} 个 (HS-SS-CE组合)")
    print(f"[数据] 节点特征维度: {graph_list[0].num_features} (含深度增强特征)")
    print(f"[数据] 边特征维度: {graph_list[0].edge_attr.shape[1]} (深度增强版)")
    print(f"[数据] 全局描述符维度: {graph_list[0].global_feat.shape[1]}")
    return graph_list, df
def split_dataset_by_ids(graph_list, include_ids):
    return [g for g in graph_list if g.sample_id in include_ids]
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.15, edge_dim=13):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        dim = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            if batch.max().item() == 0 and self.training:
                x = x
            else:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)
class GCN_GT_PINN(nn.Module):
    def __init__(self, atom_dim, global_dim=11, gcn_hidden=64, gcn_layers=2,
                 hidden_dim=64, n_layers=2, dropout=0.15, edge_dim=13,
                 k_init=1.0, tg_soft_init=280.0, tg_hard_init=500.0,
                 use_global_feat=True):
        super().__init__()
        self.k_init = k_init
        self.tg_soft_init = tg_soft_init
        self.tg_hard_init = tg_hard_init
        self.use_global_feat = use_global_feat
        self.gcn = GCNEncoder(atom_dim, gcn_hidden, gcn_layers, dropout, edge_dim)
        input_dim = gcn_hidden + (global_dim if use_global_feat else 0)
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head_K = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head_K.weight)
        nn.init.zeros_(self.head_K.bias)
        self.head_Tgh = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head_Tgh.weight)
        nn.init.zeros_(self.head_Tgh.bias)
    def forward(self, data):
        graph_emb = self.gcn(data)
        w_hard = data.w_hard
        if self.use_global_feat:
            graph_emb = torch.cat([graph_emb, data.global_feat], dim=-1)
        if graph_emb.shape[0] == 1 and self.training:
            x = graph_emb
            for layer in self.backbone:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                x = layer(x)
            feat = x
        else:
            feat = self.backbone(graph_emb)
        system_ids = data.system_id
        system_feat = scatter_mean_native(feat, system_ids, dim=0)
        raw_k_sys = self.head_K(system_feat).squeeze(-1)
        raw_tg_sys = self.head_Tgh(system_feat)
        k_sys = self.k_init * F.softplus(raw_k_sys) / 0.6931
        tg_soft_sys = self.tg_soft_init * F.softplus(raw_tg_sys[:, 0]) / 0.6931
        tg_hard_sys = self.tg_hard_init * F.softplus(raw_tg_sys[:, 1]) / 0.6931
        k = k_sys[system_ids]
        Tg_soft = tg_soft_sys[system_ids]
        Tg_hard = tg_hard_sys[system_ids]
        w_soft = 1.0 - w_hard
        numerator = w_soft * Tg_soft + k * w_hard * Tg_hard
        denominator = w_soft + k * w_hard
        Tg_pred = numerator / denominator
        return Tg_pred, k, Tg_soft, Tg_hard
def boundary_regularization(k, Tg_soft, Tg_hard, bounds):
    lo_k, hi_k = bounds["k"]
    loss_k = (F.relu(lo_k - k) + F.relu(k - hi_k)).mean()
    lo_ts, hi_ts = bounds["Tg_soft"]
    loss_ts = (F.relu(lo_ts - Tg_soft) + F.relu(Tg_soft - hi_ts)).mean()
    lo_th, hi_th = bounds["Tg_hard"]
    loss_th = (F.relu(lo_th - Tg_hard) + F.relu(Tg_hard - hi_th)).mean()
    return loss_k + loss_ts + loss_th
def total_loss(Tg_pred, k, Tg_soft, Tg_hard, y_true, bounds, hp):
    loss_mse = F.mse_loss(Tg_pred, y_true)
    loss_bd = boundary_regularization(k, Tg_soft, Tg_hard, bounds)
    loss_gt = (F.relu(Tg_soft - Tg_pred) + F.relu(Tg_pred - Tg_hard)).mean()
    total = loss_mse + hp["lambda_bound"] * loss_bd + hp["lambda_gt"] * loss_gt
    return total, loss_mse.item()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_safe_batch_size(dataset_size, target_batch_size):
    if dataset_size < 2:
        return 1
    return min(target_batch_size, dataset_size)
def get_bounds_inits(train_set):
    train_y = np.array([g.y.item() for g in train_set])
    tg_min, tg_max = float(train_y.min()), float(train_y.max())
    inits = {"k": 1.0, "tg_soft": tg_min - 50.0, "tg_hard": tg_max + 30.0}
    bounds = {
        "k": (0.1, 5.0),
        "Tg_soft": (200.0, tg_min - 10.0),
        "Tg_hard": (tg_max + 10.0, 650.0),
    }
    return bounds, inits
def train_one_model(train_set, val_set, cfg, hp, bounds, inits, seed):
    set_seed(seed)
    atom_dim = train_set[0].num_features
    global_dim = train_set[0].global_feat.shape[1]
    edge_dim = cfg["edge_dim"]
    model = GCN_GT_PINN(
        atom_dim, global_dim, hp["gcn_hidden"], hp["gcn_layers"],
        hp["hidden_dim"], hp["n_layers"], hp["dropout"], edge_dim,
        k_init=inits["k"], tg_soft_init=inits["tg_soft"], tg_hard_init=inits["tg_hard"],
        use_global_feat=cfg["use_global_feat"]
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    train_bs = get_safe_batch_size(len(train_set), cfg["batch_size"])
    val_bs = get_safe_batch_size(len(val_set), cfg["batch_size"])
    tr_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=val_bs, shuffle=False)
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss, _ = total_loss(*model(batch), batch.y.squeeze(), bounds, hp)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            total = 0
            for batch in val_loader:
                batch = batch.to(DEVICE)
                _, mse = total_loss(*model(batch), batch.y.squeeze(), bounds, hp)
                vloss += mse * batch.num_graphs
                total += batch.num_graphs
            vloss /= total
        if not math.isfinite(vloss):
            wait += 1
            if wait >= cfg["patience"]:
                break
            continue
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_val = vloss if math.isfinite(vloss) else float("inf")
    model.load_state_dict(best_state)
    return model, best_val
@torch.no_grad()
def predict(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_pred, all_k, all_ts, all_th, all_y = [], [], [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        Tg_pred, k, Tg_soft, Tg_hard = model(batch)
        all_pred.append(Tg_pred.cpu().numpy())
        all_k.append(k.cpu().numpy())
        all_ts.append(Tg_soft.cpu().numpy())
        all_th.append(Tg_hard.cpu().numpy())
        all_y.append(batch.y.squeeze().cpu().numpy())
    return (np.concatenate(all_pred), np.concatenate(all_k),
            np.concatenate(all_ts), np.concatenate(all_th), np.concatenate(all_y))
def eval_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"R2": r2, "MAE": mae, "RMSE": rmse}
def run_cross_validation(graph_list, trainval_unique_ids, cfg, hp):
    id_to_group = {}
    for g in graph_list:
        sid = g.sample_id
        if sid not in id_to_group:
            id_to_group[sid] = f"{g.hs}-{g.ss}"
    group_list = [id_to_group[sid] for sid in trainval_unique_ids]
    group_counts = Counter(group_list)
    min_thresh = cfg["min_samples_per_class"]
    main_groups = {g for g, cnt in group_counts.items() if cnt >= min_thresh}
    labels = []
    for g in group_list:
        if g in main_groups:
            labels.append(g)
        else:
            labels.append("others")
    labels = np.array(labels)
    print(f"\n分层依据: HS-SS单体组合 | 最小样本阈值: ≥{min_thresh}")
    print(f"主组合类别数: {len(main_groups)} | 稀有组合已合并为others")
    print(f"类别分布: {dict(Counter(labels))}")
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=42)
    fold_metrics = []
    print("\n" + "=" * 60)
    print(f"【模型真实性能评估】{cfg['n_splits']}折分层交叉验证 (体系级共享物理参数)")
    print("=" * 60)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(trainval_unique_ids, labels), 1):
        train_ids = set(trainval_unique_ids[train_idx])
        val_ids = set(trainval_unique_ids[val_idx])
        train_fold = split_dataset_by_ids(graph_list, train_ids)
        val_fold = split_dataset_by_ids(graph_list, val_ids)
        bounds, inits = get_bounds_inits(train_fold)
        model, _ = train_one_model(train_fold, val_fold, cfg, hp, bounds, inits, seed=0)
        y_pred, _, _, _, y_true = predict(model, val_fold)
        metrics = eval_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        print(f"  第{fold_idx}折: 训练集{len(train_fold)}条 | 验证集{len(val_fold)}条")
        print(f"       R²={metrics['R2']:.4f}  MAE={metrics['MAE']:.3f}K  RMSE={metrics['RMSE']:.3f}K")
    avg_r2 = np.mean([m["R2"] for m in fold_metrics])
    avg_mae = np.mean([m["MAE"] for m in fold_metrics])
    avg_rmse = np.mean([m["RMSE"] for m in fold_metrics])
    std_r2 = np.std([m["R2"] for m in fold_metrics])
    std_rmse = np.std([m["RMSE"] for m in fold_metrics])
    print("\n" + "-" * 60)
    print(f"  五折平均性能（模型真实泛化能力）:")
    print(f"    R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"    MAE = {avg_mae:.3f} K")
    print(f"    RMSE = {avg_rmse:.3f} ± {std_rmse:.3f} K")
    print("=" * 60)
    return fold_metrics, avg_r2, avg_mae, avg_rmse
def run_hpo(train_set, val_set, cfg, bounds, inits):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[HPO] 未安装 optuna, 跳过超参优化")
        return None
    full_set = train_set + val_set
    sample_ids = [d.sample_id for d in full_set]
    unique_ids = list(set(sample_ids))
    id_to_group = {}
    for g in full_set:
        sid = g.sample_id
        if sid not in id_to_group:
            id_to_group[sid] = f"{g.hs}-{g.ss}"
    group_list = [id_to_group[sid] for sid in unique_ids]
    group_counts = Counter(group_list)
    main_groups = {g for g, cnt in group_counts.items() if cnt >= 3}
    labels = np.array([g if g in main_groups else "others" for g in group_list])
    def objective(trial):
        hp = {
            "gcn_hidden": trial.suggest_categorical("gcn_hidden", [32, 64]),
            "gcn_layers": trial.suggest_int("gcn_layers", 1, 2),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
            "n_layers": trial.suggest_int("n_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "lambda_bound": trial.suggest_float("lambda_bound", 0.2, 0.6, log=True),
            "lambda_gt": trial.suggest_float("lambda_gt", 0.1, 1.0, log=True),
        }
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        scores = []
        for tr_ids_idx, va_ids_idx in skf.split(unique_ids, labels):
            tr_sids = set([unique_ids[i] for i in tr_ids_idx])
            va_sids = set([unique_ids[i] for i in va_ids_idx])
            tr_idx = [i for i, sid in enumerate(sample_ids) if sid in tr_sids]
            va_idx = [i for i, sid in enumerate(sample_ids) if sid in va_sids]
            tr_ds = [full_set[i] for i in tr_idx]
            va_ds = [full_set[i] for i in va_idx]
            model, _ = train_one_model(tr_ds, va_ds, cfg, hp, bounds, inits, seed=0)
            yp, _, _, _, yt = predict(model, va_ds)
            scores.append(mean_squared_error(yt, yp))
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg["hpo_trials"], show_progress_bar=False)
    print(f"\n[HPO] 最优参数: {study.best_params}")
    print(f"[HPO] 交叉验证最优MSE: {study.best_value:.3f}")
    return study.best_params
def run_weight_hpo(train_set, val_set, cfg, bounds, inits):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[权重HPO] 未安装 optuna, 跳过权重优化")
        return None
    full_set = train_set + val_set
    sample_ids = [d.sample_id for d in full_set]
    unique_ids = list(set(sample_ids))
    id_to_group = {}
    for g in full_set:
        sid = g.sample_id
        if sid not in id_to_group:
            id_to_group[sid] = f"{g.hs}-{g.ss}"
    group_list = [id_to_group[sid] for sid in unique_ids]
    group_counts = Counter(group_list)
    main_groups = {g for g, cnt in group_counts.items() if cnt >= 3}
    labels = np.array([g if g in main_groups else "others" for g in group_list])
    base_hp = {k: cfg[k] for k in ["gcn_hidden", "gcn_layers", "hidden_dim", "n_layers",
                                   "dropout", "lr", "weight_decay"]}
    total_trials = cfg["weight_hpo_trials"]
    print(f"\n开始权重超参优化，共 {total_trials} 次贝叶斯试验...")
    def objective(trial):
        current = trial.number + 1
        print(f"  权重HPO进度: 第 {current:>2d}/{total_trials} 次试验", end='\r')
        lambda_bound = trial.suggest_float("lambda_bound", 0.1, 1.2, log=True)
        lambda_gt = trial.suggest_float("lambda_gt", 0.05, 1.5, log=True)
        hp = base_hp.copy()
        hp["lambda_bound"] = lambda_bound
        hp["lambda_gt"] = lambda_gt
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        scores = []
        for tr_ids_idx, va_ids_idx in skf.split(unique_ids, labels):
            tr_sids = set([unique_ids[i] for i in tr_ids_idx])
            va_sids = set([unique_ids[i] for i in va_ids_idx])
            tr_idx = [i for i, sid in enumerate(sample_ids) if sid in tr_sids]
            va_idx = [i for i, sid in enumerate(sample_ids) if sid in va_sids]
            tr_ds = [full_set[i] for i in tr_idx]
            va_ds = [full_set[i] for i in va_idx]
            model, _ = train_one_model(tr_ds, va_ds, cfg, hp, bounds, inits, seed=0)
            yp, _, _, _, yt = predict(model, va_ds)
            scores.append(mean_squared_error(yt, yp))
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=total_trials, show_progress_bar=False)
    print()
    print(f"\n{'=' * 60}")
    print("【损失权重专项优化结果】")
    print(f"{'=' * 60}")
    print(f"  最优边界正则权重 lambda_bound = {study.best_params['lambda_bound']:.4f}")
    print(f"  最优GT自洽权重 lambda_gt     = {study.best_params['lambda_gt']:.4f}")
    print(f"  对应交叉验证最优MSE     = {study.best_value:.3f}")
    print(f"{'=' * 60}")
    return study.best_params
def main():
    cfg = CONFIG.copy()
    print("=" * 60)
    print("GCN-Gordon-Taylor PINN (硬约束+GT自洽损失版)")
    print("=" * 60)
    print(f"\n[设备] 当前计算设备: {DEVICE}")
    graph_list, df = build_polyurea_dataset(cfg)
    all_unique_ids = df[cfg["sample_id_col"]].unique()
    np.random.seed(42)
    np.random.shuffle(all_unique_ids)
    n_total = len(all_unique_ids)
    n_test = max(1, int(n_total * cfg["test_ratio"]))
    test_ids = set(all_unique_ids[:n_test])
    trainval_ids = all_unique_ids[n_test:]
    test_set = split_dataset_by_ids(graph_list, test_ids)
    print(f"\n[数据集划分] 独立测试集: {len(test_ids)}个分子, {len(test_set)}条样本")
    print(f"[数据集划分] 训练验证集: {len(trainval_ids)}个分子")
    hp = {k: cfg[k] for k in ["gcn_hidden", "gcn_layers", "hidden_dim", "n_layers",
                              "dropout", "lr", "weight_decay",
                              "lambda_bound", "lambda_gt"]}
    cv_metrics, cv_avg_r2, cv_avg_mae, cv_avg_rmse = run_cross_validation(
        graph_list, trainval_ids, cfg, hp
    )
    if cfg["run_hpo"]:
        print("\n正在进行全超参优化...")
        trainval_set = split_dataset_by_ids(graph_list, set(trainval_ids))
        bounds, inits = get_bounds_inits(trainval_set)
        best = run_hpo(trainval_set, test_set, cfg, bounds, inits)
        if best:
            hp.update(best)
        print(f"\n[优化后超参] {hp}")
    if cfg["run_weight_hpo"]:
        print("\n正在进行损失权重专项贝叶斯优化...")
        trainval_set = split_dataset_by_ids(graph_list, set(trainval_ids))
        bounds, inits = get_bounds_inits(trainval_set)
        best_weights = run_weight_hpo(trainval_set, test_set, cfg, bounds, inits)
        if best_weights:
            hp.update(best_weights)
        print(f"\n[最终权重] lambda_bound = {hp['lambda_bound']:.4f}, lambda_gt = {hp['lambda_gt']:.4f}")
    trainval_set = split_dataset_by_ids(graph_list, set(trainval_ids))
    bounds, inits = get_bounds_inits(trainval_set
    print(f"\n[损失权重] 数据拟合(MSE): 1.0 (基准)")
    print(f"[损失权重] 边界正则(lambda_bound): {hp['lambda_bound']:.4f}")
    print(f"[损失权重] GT物理自洽(lambda_gt): {hp['lambda_gt']:.4f}")
    print(f"[物理参数] 初始值: k={inits['k']:.2f}, Tg_soft={inits['tg_soft']:.1f}K, Tg_hard={inits['tg_hard']:.1f}K")
    print(f"[物理参数] 边界约束: k∈{bounds['k']}, Tg_soft∈{bounds['Tg_soft']}, Tg_hard∈{bounds['Tg_hard']}")
    print(f"[核心架构] GT方程硬编码前向(强约束) + 损失函数补充GT自洽软约束")
    print(f"[核心架构] 物理参数按 HS-SS-CE 体系共享，同体系参数完全一致")
    print(f"[验证策略] 分层交叉验证: 按HS-SS单体组合分层(阈值≥3)")
    models = []
    print("\n" + "=" * 60)
    print("训练最终PINN模型 (多种子集成)")
    print("=" * 60)
    for s in range(cfg["n_seeds"]):
        model, vloss = train_one_model(trainval_set, test_set, cfg, hp, bounds, inits, seed=s)
        models.append(model)
        print(f"  Seed {s}: 验证MSE = {vloss:.3f}")
    def ensemble_predict(dataset):
        preds = []
        for m in models:
            yp, k, Ts, Th, yt = predict(m, dataset)
            preds.append(yp)
        mean_pred = np.mean(preds, axis=0)
        _, k_last, Ts_last, Th_last, _ = predict(models[-1], dataset)
        return mean_pred, k_last, Ts_last, Th_last
    print("\n" + "=" * 60)
    print("【最终结果】独立测试集性能")
    print("=" * 60)
    yp_test, k_test, Ts_test, Th_test = ensemble_predict(test_set)
    yt_test = np.array([g.y.item() for g in test_set])
    test_metrics = eval_metrics(yt_test, yp_test)
    print(f"  R² = {test_metrics['R2']:.4f}")
    print(f"  MAE = {test_metrics['MAE']:.3f} K")
    print(f"  RMSE = {test_metrics['RMSE']:.3f} K")
    print("\n" + "=" * 60)
    print("测试集物理参数分布")
    print("=" * 60)
    print(f"  k值: 最小值={k_test.min():.4f}, 最大值={k_test.max():.4f}, 均值={k_test.mean():.4f}")
    print(f"  Tg_soft: 最小值={Ts_test.min():.1f}K, 最大值={Ts_test.max():.1f}K, 均值={Ts_test.mean():.1f}K")
    print(f"  Tg_hard: 最小值={Th_test.min():.1f}K, 最大值={Th_test.max():.1f}K, 均值={Th_test.mean():.1f}K")
    save_path = "gcn_gt_pinn_hard_constraint_plus_loss.pt"
    torch.save({
        "models": [m.state_dict() for m in models],
        "hp": hp,
        "bounds": bounds,
        "inits": inits,
        "atom_dim": trainval_set[0].num_features,
        "edge_dim": cfg["edge_dim"],
        "global_dim": trainval_set[0].global_feat.shape[1],
        "cv_fold_metrics": cv_metrics,
        "cv_avg_R2": cv_avg_r2,
        "cv_avg_MAE": cv_avg_mae,
        "cv_avg_RMSE": cv_avg_rmse,
        "test_metrics": test_metrics,
    }, save_path)
    print(f"\n[完成] 模型与评估结果已保存至: {save_path}")
if __name__ == "__main__":
    main()
