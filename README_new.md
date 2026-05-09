# 训练与数据适配文档（RadioMapSeer / BRATLab Radiomap-Data / UrbanRadio3D）

本文档说明：当你拥有 **RadioMapSeer、BRATLab Radiomap-Data、UrbanRadio3D** 这三类数据后，应该如何组织数据、编写/改造训练代码，使其能在同一套训练入口下完成训练、验证、评测与 Demo 产出。

本仓库当前核心代码位置：

- 训练入口：[train.py](file:///d:/RME-GAN/train.py)
- 评测/Demo 入口：[test.py](file:///d:/RME-GAN/test.py)
- RadioMapSeer 数据加载器示例：[loaders.py](file:///d:/RME-GAN/lib/loaders.py)
- 模型结构（RadioWNet）：[modules.py](file:///d:/RME-GAN/lib/modules.py)

## 1) 统一样本协议（你要先定死的标准）

无论来自哪个数据集，Adapter 最终都要输出统一格式：

- 输入 `x`：`torch.FloatTensor`，形状 `[C, H, W]`
- 标签 `y`：`torch.FloatTensor`，形状 `[1, H, W]`
- 元信息 `meta`（可选）：字典（`scene_type / area_id / cell_id / has_env / grid_size / meters_per_pixel ...`）

强烈建议你把 `C` 的通道语义固定为以下 4 类（与本仓库 RadioMapSeer 基线一致）：

1. `buildings_or_env`：建筑物占用图/高度图/土地覆盖（缺失则全 0）
2. `tx_mask`：发射点位栅格（缺失则无法做按站点预测）
3. `sparse_samples`：稀疏测量点（从真值采样得到；没实测也可从 GT 抽样模拟）
4. `prior`：先验图（可选；例如简化路径损耗模型拟合出的 `genImg`）

## 2) 训练代码应该怎么写（推荐的最小改造方式）

你现在的 [train.py](file:///d:/RME-GAN/train.py) 已经能跑通 RadioMapSeer。要支持三个数据集，建议按“最小侵入”做：

本项目建议拆成 **3 个核心模块**（加 1 个可选模块）：

1. **Dataset Adapter 模块（数据适配层）**：为每个数据集实现一个 Adapter，输出统一的 `(x, y, meta)`。
2. **Dataset Factory 模块（数据集工厂层）**：根据 `--dataset/--dataset_dir` 构造 train/val/test 的 Dataset。
3. **Trainer 模块（训练器/训练循环）**：训练/验证/保存权重/记录指标；只依赖统一的 `(x, y)`，不写数据集特判。
4. **Multi-dataset Sampler 模块（可选，混训采样层）**：当需要三数据集混训时，负责在多个 DataLoader 间按权重采样 batch，并分别统计验证指标。

换句话说：训练代码的变化应集中在“数据加载与统一化”，不要让训练循环里出现任何“if 数据集是 X”的逻辑。

## 3) 三个数据集的 Adapter 分别怎么做

### 3.1 RadioMapSeer（最容易对接）

你可以直接复用当前的 `RadioUNet_s`（见 [loaders.py](file:///d:/RME-GAN/lib/loaders.py)）：

- `x` 通道（默认 4）：`[buildings, Tx_mask, sparse_samples, genImg]`
- `y`：`image_gain`（注意：是 0~256 归一化，不等同于 dB）

你必须额外明确：

- `y` 的物理意义（pathloss/RSRP/接收功率/归一化 gain）
- 评测口径的 dB 映射（`test.py` 已支持线性映射参数）

### 3.2 BRATLab Radiomap-Data（关键在“组织成统一栅格样本”）

BRATLab 数据常见形态是：

- 点集：`(x, y, power)` 或矩阵：`power[H,W]`
- 可能只有全网合成覆盖，不一定能拆到单 cell

Adapter 实现步骤建议：

1. 选择预测目标（`y`）
   - 如果能按 cell 拆分：每个样本对应一个 `cell_id` 的覆盖图
   - 否则：把“全网合成覆盖图”作为 `y`
2. 构建输入通道（`x`）
   - `tx_mask`：站点位置栅格化（全网合成场景也建议把所有站点都画进去）
   - `sparse_samples`：从 `y` 随机抽点生成稀疏测量（或用你们的真实测量点）
   - `buildings_or_env`：若数据集不提供环境，就先用全 0（效果上限会受限，但能跑通流程）
   - `prior`：可先不做；或用简化路径损耗模型做一个粗先验
3. 统一网格与分辨率
   - 将原始坐标/矩阵映射到统一 `H×W`（例如 256×256）
   - 在 `meta` 里记录 `meters_per_pixel` 与区域范围

### 3.3 UrbanRadio3D（先做 2D baseline，再做 3D）

UrbanRadio3D 这类数据往往是多高度层/多指标（pathloss/DoA/ToA 等）。推荐两阶段落地：

- 阶段 A（推荐，先跑通）
  - 选择一个固定高度层（例如 1.5m 对应的层 index）
  - 用该层的 `pathloss` 作为 `y`
  - 环境输入仍为 2D（建筑投影/高度图）
  - 模型仍用现有 2D `RadioWNet`
- 阶段 B（需要重构模型）
  - 改 3D 卷积或多高度层多任务输出
  - 显存与效率成本更高，建议在阶段 A 指标稳定后再做

## 4) 预处理与归一化（必须统一，否则混训会直接失效）

跨数据集训练要先统一协议：

- **空间范围**：一个样本代表的真实面积（例如 1 km²）要固定或写入 meta
- **分辨率**：统一到同一 `H×W`（例如 256×256），并固定 `meters_per_pixel`
- **通道语义**：`C` 个通道在所有数据集里含义一致
- **标签单位**：
  - 训练可以用归一化后的数值，但
  - 评测必须统一到 dB/dBm，并能输出 MAE/RMSE/P95/≤7 dB 占比

## 5) 多数据集训练策略（从简单到复杂）

推荐顺序（更稳妥）：

1. 只用 RadioMapSeer：先把训练、保存权重、评测指标、Demo 图片全部跑通
2. 迁移学习：
   - 用 RadioMapSeer 预训练权重，在 BRATLab 上 fine-tune
   - 或在 UrbanRadio3D（2D 化后的高度层）上 fine-tune
3. 三数据集混训（最后做）：
   - 每个数据集独立 DataLoader
   - 按权重采样数据集输出 batch
   - 验证集必须分数据集统计指标（否则会出现“平均变好，但每个都不达标”的情况）

## 6) 训练入口参数（建议你这样设计）

本仓库训练入口已经有 `--dataset_dir --setup --epochs --batch_size ...`。

要支持三个数据集，建议增加：

- `--dataset`：`radiomapseer|bratlab|urbanradio3d`
- `--grid_size`：如 256
- `--meters_per_pixel`：如 4.0（配合 1 km²）
- `--height_layer`：UrbanRadio3D 使用
- `--scene_type`：可选（用于分场景训练/评测）

评测继续用 [test.py](file:///d:/RME-GAN/test.py)：

- `--weights` 指定权重
- `--db_min/--db_max` 或 `--value_scale/--value_offset` 指定 dB 映射
- 输出 MAE/RMSE/P90/P95、≤7 dB 占比与推理耗时

## 7) 最小“训练代码完成标准”（按这个验收）

- 新增数据集时：只新增 Adapter + 在 Factory 注册；训练循环不改
- 三个数据集都能：训练、保存权重、评测输出指标（含 dB 口径）与 Demo 图
- 能在固定的 1 km²、≥20 Tx 场景上跑推理并输出耗时（对齐 ≤30 分钟要求）
