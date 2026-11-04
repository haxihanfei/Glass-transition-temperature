import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)
class Config:
    hidden_channels = 80
    out_channels = 48
    global_feat_dim = 25
    seg_hidden = 32
    dropout = 0.35
    epochs = 300
    batch_size = 16
    lr = 8e-4
    weight_decay = 2e-4
    alpha = 0.15  
    early_stop_patience = 60
    grad_clip_norm = 1.0
    seg_feat_start = 27
    seg_feat_end = 30
    num_seg_types = 3
    warmup_epochs = 10
cfg = Config()
def extract_global_features(dataset):
    for data in dataset:
        x = data.x
        elem_ratio = x[:, 0:5].mean(dim=0)
        degree_ratio = x[:, 5:10].mean(dim=0)
        seg_ratio = x[:, cfg.seg_feat_start:cfg.seg_feat_end].mean(dim=0)
        hb_ratio = x[:, 30:34].mean(dim=0)
        aromatic_ratio = x[:, 14:15].mean()
        global_feat = torch.cat([
            elem_ratio, degree_ratio, seg_ratio, hb_ratio, aromatic_ratio.unsqueeze(0),
            formal_charge_mean.unsqueeze(0), radical_e_mean.unsqueeze(0), implicit_h_ratio
        ]).float()
        data.global_feat = global_feat.unsqueeze(0)
    return dataset
def segment_center_loss(node_emb, seg_labels):
    num_seg = cfg.num_seg_types
    centers = []
    intra_loss = 0.0
    for i in range(num_seg):
        mask = seg_labels == i
        if mask.sum() > 0:
            center = node_emb[mask].mean(dim=0)
            centers.append(center)
            intra_loss += F.mse_loss(node_emb[mask], center.unsqueeze(0).expand(mask.sum(), -1))
        else:
            centers.append(torch.zeros(node_emb.shape[1], device=node_emb.device))
    inter_loss = 0.0
    count = 0
    for i in range(num_seg):
        for j in range(i + 1, num_seg):
            dist = F.pairwise_distance(centers[i].unsqueeze(0), centers[j].unsqueeze(0))
            inter_loss += 1.0 / (dist + 1e-6)
            count += 1
    if count > 0:
        inter_loss = inter_loss / count
    return intra_loss + 0.5 * inter_loss
class GINE_Global_Fusion(nn.Module):
    def __init__(self, in_channels, edge_dim, global_dim):
        super().__init__()
        self.node_proj = nn.Linear(in_channels, cfg.hidden_channels)
        self.edge_proj = nn.Linear(edge_dim, cfg.hidden_channels)
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
                nn.ReLU(),
                nn.Linear(cfg.hidden_channels, cfg.hidden_channels)
            ),
            train_eps=True,
            edge_dim=cfg.hidden_channels
        )
        self.bn1 = nn.BatchNorm1d(cfg.hidden_channels)
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
                nn.ReLU(),
                nn.Linear(cfg.hidden_channels, cfg.hidden_channels)
            ),
            train_eps=True,
            edge_dim=cfg.hidden_channels
        )
        self.bn2 = nn.BatchNorm1d(cfg.hidden_channels)
        self.conv3 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(cfg.hidden_channels, cfg.out_channels),
                nn.ReLU(),
                nn.Linear(cfg.out_channels, cfg.out_channels)
            ),
            train_eps=True,
            edge_dim=cfg.hidden_channels
        )
        self.bn3 = nn.BatchNorm1d(cfg.out_channels)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(cfg.out_channels * 2 + 32, cfg.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_channels // 2, 1)
        )
        self.seg_cls = nn.Linear(cfg.out_channels, cfg.num_seg_types)
    def forward(self, data):
        x, edge_index, edge_attr, batch, global_feat = (
            data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat
        )
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        node_emb = F.relu(x)
        mol_mean = global_mean_pool(node_emb, batch)
        mol_max = global_max_pool(node_emb, batch)
        graph_emb = torch.cat([mol_mean, mol_max], dim=1)
        global_emb = self.global_mlp(global_feat)
        fused_emb = torch.cat([graph_emb, global_emb], dim=1)
        pred = self.predictor(fused_emb)
        return pred, node_emb
def train_epoch(model, loader, optimizer, scaler_mean, scaler_std, device):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_seg = 0.0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, node_emb = model(data)
        y = data.y.view(-1, 1)
        y_norm = (y - scaler_mean) / scaler_std
        mse_loss = F.mse_loss(pred, y_norm)
        seg_labels = data.x[:, cfg.seg_feat_start:cfg.seg_feat_end].argmax(dim=1)
        seg_loss = segment_center_loss(node_emb, seg_labels)
        loss = mse_loss + cfg.alpha * seg_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        loss.backward()
        optimizer.step()
        bs = data.num_graphs
        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_seg += seg_loss.item() * bs
        total_samples += bs
    return total_loss / total_samples, total_mse / total_samples, total_seg / total_samples
@torch.no_grad()
def eval_model(model, loader, scaler_mean, scaler_std, device, use_tta=True):
    model.eval()
    preds = []
    trues = []
    sample_ids = []
    for data in loader:
        data = data.to(device)
        pred_norm, _ = model(data)
        pred_original = (pred_norm * scaler_std + scaler_mean).cpu().numpy().flatten()
        true_original = data.y.cpu().numpy().flatten()
        preds.append(pred_original)
        trues.append(true_original)
        sample_ids.extend(data.sample_id)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if use_tta:
        unique_ids = np.unique(sample_ids)
        agg_preds = []
        agg_trues = []
        for sid in unique_ids:
            mask = np.array(sample_ids) == sid
            agg_preds.append(preds[mask].mean())
            agg_trues.append(trues[mask].mean())
        preds = np.array(agg_preds)
        trues = np.array(agg_trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    return rmse, r2
def get_lr_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def group_kfold_cv(dataset, n_splits=10):
    sample_ids = np.array([d.sample_id for d in dataset])
    unique_ids = np.unique(sample_ids)
    print(f"原始分子数：{len(unique_ids)}，增强后总样本数：{len(dataset)}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rmse = []
    fold_r2 = []
    for fold, (train_ids_idx, test_ids_idx) in enumerate(kf.split(unique_ids)):
        print(f"\n{'=' * 60}")
        print(f"第 {fold + 1} 折交叉验证")
        print(f"{'=' * 60}")
        train_ids = set(unique_ids[train_ids_idx])
        test_ids = set(unique_ids[test_ids_idx])
        train_idx = [i for i, sid in enumerate(sample_ids) if sid in train_ids]
        test_idx = [i for i, sid in enumerate(sample_ids) if sid in test_ids]
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        train_y = np.array([d.y.item() for d in train_dataset]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(train_y)
        scaler_mean = torch.tensor(scaler.mean_[0], dtype=torch.float32, device=device)
        scaler_std = torch.tensor(scaler.scale_[0], dtype=torch.float32, device=device)
        model = GINE_Global_Fusion(
            in_channels=dataset[0].num_features,
            edge_dim=dataset[0].num_edge_features,
            global_dim=cfg.global_feat_dim
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = get_lr_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
        best_rmse = float('inf')
        best_r2 = -float('inf')
        patience_counter = 0
        best_state = None
        for epoch in tqdm(range(cfg.epochs), desc=f"Fold {fold + 1}"):
            train_loss, train_mse, train_seg = train_epoch(
                model, train_loader, optimizer, scaler_mean, scaler_std, device
            )
            val_rmse, val_r2 = eval_model(model, test_loader, scaler_mean, scaler_std, device)
            scheduler.step()
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_r2 = val_r2
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    print(f"\n早停触发，最佳轮次：{epoch - cfg.early_stop_patience + 1}")
                    break
            if (epoch + 1) % 50 == 0:
                print(
                    f"\nEpoch {epoch + 1:3d} | 总损失:{train_loss:.4f} | MSE:{train_mse:.4f} | 链段正则:{train_seg:.4f}")
                print(f"          验证集 RMSE:{val_rmse:.4f} | R²:{val_r2:.4f}")
        fold_rmse.append(best_rmse)
        fold_r2.append(best_r2)
        print(f"\n第 {fold + 1} 折最佳结果：RMSE = {best_rmse:.4f} | R² = {best_r2:.4f}")
    print(f"\n{'=' * 60}")
    print(f"{n_splits}折交叉验证最终结果")
    print(f"{'=' * 60}")
    print(f"RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
    print(f"R²:   {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
    return fold_rmse, fold_r2
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("正在加载数据集...")
    dataset = torch.load(cfg.data_path, weights_only=False)
    print("正在计算全局分子描述符...")
    dataset = extract_global_features(dataset)
    print(f"节点特征维度: {dataset[0].num_features} | 边特征维度: {dataset[0].num_edge_features}")
    print(f"全局特征维度: {dataset[0].global_feat.shape[1]}")
    group_kfold_cv(dataset, n_splits=10)
