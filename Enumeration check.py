import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
df = pd.read_csv(file_path)
print(f"总样本数：{len(df)}")
print(f"原始样本数：{len(df[df['is_original'] == 1])}")
print(f"枚举样本数：{len(df[df['is_original'] == 0])}")
original_df = df[df['is_original'] == 1].copy()
original_smi_map = {}
for _, row in original_df.iterrows():
    sample_id = row['sample_id']
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol is None:
        print(f"警告：原始样本 {sample_id} SMILES 无法解析！")
        continue
    canonical_smi = Chem.MolToSmiles(mol)  # 统一转为规范SMILES作为基准
    heavy_atom_num = mol.GetNumHeavyAtoms()
    original_smi_map[sample_id] = {
        "canonical_smi": canonical_smi,
        "heavy_atom_num": heavy_atom_num,
        "mol_weight": Descriptors.MolWt(mol)
    }
valid_count = 0
structure_correct_count = 0
error_samples = []
for idx, row in df.iterrows():
    smi = row['SMILES']
    sample_id = row['sample_id']
    augment_id = row['augment_id']
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        error_samples.append({
            "augment_id": augment_id,
            "error_type": "无法解析"
        })
        continue
    valid_count += 1
    if row['is_original'] == 0:
        enum_canonical = Chem.MolToSmiles(mol)
        # 与对应原始样本的规范SMILES对比
        if sample_id in original_smi_map:
            ref_canonical = original_smi_map[sample_id]['canonical_smi']
            if enum_canonical == ref_canonical:
                structure_correct_count += 1
            else:
                error_samples.append({
                    "augment_id": augment_id,
                    "error_type": "结构不一致",
                    "原始规范SMILES": ref_canonical,
                    "枚举规范SMILES": enum_canonical
                })
print("\n" + "=" * 50)
print("校验报告")
print("=" * 50)
print(f"可正常解析样本数：{valid_count} / {len(df)}")
print(f"解析失败数：{len(df) - valid_count}")
print(f"枚举样本结构一致数：{structure_correct_count} / {len(df[df['is_original'] == 0])}")
print(f"结构异常样本数：{len([e for e in error_samples if e['error_type'] == '结构不一致'])}")
if len(error_samples) > 0:
    print("\n异常样本明细：")
    for e in error_samples:
        print(f"- {e['augment_id']}：{e['error_type']}")
else:
    print("\n✅ 所有枚举样本均合法且与原始结构完全一致，可直接用于模型训练。")
