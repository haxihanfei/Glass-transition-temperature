import pandas as pd
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')
df_raw = pd.read_csv(file_path)
df_raw = df_raw[['HS', 'SS', 'CE', 'SMILES', 'TG']].copy()
df_raw['SMILES'] = df_raw['SMILES'].str.strip()
df_raw['is_original'] = 1
df_raw['sample_id'] = [f"PU_{i:03d}" for i in range(len(df_raw))]
valid_mask = df_raw['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
invalid_num = len(df_raw) - valid_mask.sum()
if invalid_num > 0:
    print(f"警告：发现 {invalid_num} 个无效SMILES，已自动过滤")
    df_raw = df_raw[valid_mask].reset_index(drop=True)
print(f"原始有效样本总数: {len(df_raw)}")
print(f"硬段种类: {df_raw['HS'].unique().tolist()}")
print(f"软段种类: {df_raw['SS'].unique().tolist()}")
print(f"扩链剂种类: {df_raw['CE'].unique().tolist()}")
def enumerate_smiles(smiles_str, n_target):
    """
    对单个SMILES生成n_target个去重的非规范SMILES
    :param smiles_str: 原始规范SMILES
    :param n_target: 目标枚举数量
    :return: 枚举后的SMILES列表
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return []
    variants = set()
    max_try = n_target * 5
    for _ in range(max_try):
        rand_smi = Chem.MolToSmiles(mol, doRandom=True)
        variants.add(rand_smi)
        if len(variants) >= n_target:
            break
    return list(variants)
def get_enumerate_num(smiles_str):
    return 14
augmented_rows = []
for idx, row in df_raw.iterrows():
    original_row = row.to_dict()
    original_row['augment_id'] = f"{row['sample_id']}_orig"
    original_row['augment_type'] = 'original'
    augmented_rows.append(original_row)
    n_enum = get_enumerate_num(row['SMILES'])
    smi_variants = enumerate_smiles(row['SMILES'], n_enum)
    for i, smi in enumerate(smi_variants):
        new_row = row.to_dict()
        new_row['SMILES'] = smi
        new_row['augment_id'] = f"{row['sample_id']}_enum_{i + 1:02d}"
        new_row['augment_type'] = 'smiles_enum'
        new_row['is_original'] = 0
        augmented_rows.append(new_row)
df_aug = pd.DataFrame(augmented_rows)
df_aug = df_aug.sort_values('is_original', ascending=False).drop_duplicates(subset=['SMILES'], keep='first')
df_aug = df_aug.reset_index(drop=True)
enum_num = len(df_aug) - len(df_raw)
print(f"\n===== 增强完成 =====")
print(f"原始样本数：{len(df_raw)}")
print(f"新增枚举样本数：{enum_num}")
print(f"最终总样本数：{len(df_aug)}")
print(f"统一枚举倍数：每个原始分子生成 14 个SMILES变体")
save_path = r"C:\Users\user\Desktop\tc\shenxuxuexi\3TG\SMILES枚举增强数据集_14倍.csv"
df_aug.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"增强后数据集已保存至：{save_path}")
