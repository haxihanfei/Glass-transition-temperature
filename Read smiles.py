import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
def load_dataset(file_path):
    try:
        dataset = torch.load(file_path, weights_only=False)
        print(f"✅ 数据集加载成功！共包含 {len(dataset)} 个分子图样本\n")
        return dataset
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {file_path}，请检查路径是否正确")
        return None
    except Exception as e:
        print(f"❌ 加载失败，错误信息：{e}")
        return None
def dataset_overview(dataset):
    print("=" * 60)
    print("📊 数据集整体概览")
    print("=" * 60)
    num_samples = len(dataset)
    num_nodes_list = [data.num_nodes for data in dataset]
    num_edges_list = [data.num_edges for data in dataset]
    tg_list = [data.y.item() for data in dataset]
    print(f"总样本数：{num_samples}")
    # 修正：分母改为 len(num_nodes_list)
    print(f"节点数范围：{min(num_nodes_list)} ~ {max(num_nodes_list)}，平均 {sum(num_nodes_list)/len(num_nodes_list):.1f}")
    # 修正：分母改为 len(num_edges_list)
    print(f"边数范围：{min(num_edges_list)} ~ {max(num_edges_list)}，平均 {sum(num_edges_list)/len(num_edges_list):.1f}")
    print(f"Tg标签范围：{min(tg_list):.2f} K ~ {max(tg_list):.2f} K，平均 {sum(tg_list)/len(tg_list):.2f} K")
    print(f"单样本节点特征维度：{dataset[0].num_features}")
    print(f"单样本边特征维度：{dataset[0].num_edge_features}")
    original_count = sum(1 for data in dataset if data.is_original == 1)
    augment_count = num_samples - original_count
    print(f"原始样本数：{original_count}，枚举增强样本数：{augment_count}\n")
def show_sample_detail(dataset, index=0):
    print("=" * 60)
    print(f"🔍 第 {index+1} 个样本详细信息")
    print("=" * 60)
    data = dataset[index]
    print(f"样本ID：{data.sample_id}")
    print(f"增强ID：{data.augment_id}")
    print(f"是否原始样本：{'是' if data.is_original == 1 else '否'}")
    print(f"Tg标签值：{data.y.item():.3f} K")
    print(f"原子（节点）总数：{data.num_nodes}")
    print(f"化学键（边）总数：{data.num_edges}")
    print("\n--- 节点特征矩阵 x 形状：", data.x.shape)
    print("前3个原子的特征向量示例：")
    for i in range(min(3, data.num_nodes)):
        print(f"  原子{i}：{data.x[i].tolist()}")
    print("\n--- 边索引矩阵 edge_index 形状：", data.edge_index.shape)
    print("前5条边的连接关系（无向图，双向存储）：")
    for i in range(min(5, data.num_edges)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        print(f"  边{i}：原子{src} <--> 原子{dst}")
    print("\n--- 边特征矩阵 edge_attr 形状：", data.edge_attr.shape)
    print("前3条边的特征向量示例：")
    for i in range(min(3, data.num_edges)):
        print(f"  边{i}：{data.edge_attr[i].tolist()}")
    print()
def print_feature_desc():
    print("=" * 60)
    print("📝 特征维度说明")
    print("=" * 60)
    print("【节点特征（共27维）】")
    print("  0~4维：元素类型独热（C、N、O、S、未知）")
    print("  5~9维：原子成键度数独热（1、2、3、4、未知）")
    print("  10~13维：杂化类型独热（SP、SP2、SP3、未知）")
    print("  14维：芳香性标记（0=非芳香，1=芳香）")
    print("  15~19维：隐氢原子数独热（0、1、2、3、未知）")
    print("  20~22维：链段归属独热（软段、硬段、扩链剂）")
    print("  23~26维：氢键角色独热（无、给体、受体、兼具）")
    print("\n【边特征（共5维）】")
    print("  0~3维：键级类型独热（单键、双键、三键、芳香键）")
    print("  4维：主链标记（1=主链化学键，0=侧链/环上化学键）")
    print()
if __name__ == "__main__":
    dataset = load_dataset(PT_FILE_PATH)
    if dataset is None:
        exit()
    dataset_overview(dataset)
    print_feature_desc()
    show_sample_detail(dataset, index=0)
    print("=" * 60)
    print("💡 批量遍历示例（前5个样本的ID与Tg）")
    print("=" * 60)
    for i, data in enumerate(dataset[:5]):
        print(f"样本{i+1}：ID={data.sample_id:8s} | Tg={data.y.item():.3f} K | 原子数={data.num_nodes:3d}")

    print("\n✅ 读取完成！你可以通过 dataset[i] 访问第i个分子图对象")
