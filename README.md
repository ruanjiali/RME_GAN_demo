# 无线网络覆盖预测系统（基于 RME-GAN / Radio Map Estimation）

本仓库当前实现的是基于条件 GAN / U-Net 的无线电“覆盖图（radio map）”估计：输入为建筑物栅格、发射点位、稀疏测量点以及一个简化的路径损耗拟合图，输出为高分辨率的覆盖强度分布图（以 RadioMapSeer 数据的归一化方式表示）。

如果你的参赛目标是“大模型驱动的无线网络覆盖预测”，并且需要对路径损耗、RSRP 等关键指标与实测数据的偏差 < 7 dB、覆盖不少于 3 类场景、并满足复杂城市场景推理时延约束，那么本仓库目前只能算“一个原型/基线”，仍需要补齐数据、评测口径与工程化推理流程。

## 与比赛指标的对照结论（基于当前代码现状）

- 预测精度（< 7 dB）：当前代码默认只算 MSE/NMSE（且与像素归一化强相关），未提供“以 dB 为单位”的误差评测与阈值统计；也未明确 RSRP/路径损耗与像素值的物理映射。需要补齐 dB 口径评测与数据标定。
- 场景覆盖（≥3 类）：当前数据加载器默认面向 RadioMapSeer 单一城市场景数据组织；没有“城市高楼密集 / 郊区 / 开阔田野”三类场景的数据接口与实验脚本。需要引入多场景数据并做分场景评测。
- 计算效率（1 km² / ≥20 Tx / ≤30 min）：模型本身是卷积网络，单卡推理通常远低于 30 分钟；但当前仓库没有“面向 1 km²、≥20 发射点”的端到端批处理推理与计时基准，也没有将输入从工程数据（GIS/DTM/站点表/测量点）转换为网络输入栅格的流水线。需要补齐工程化推理与基准测试。

## 代码结构（核心源码）

- 训练入口：[train.py](file:///d:/RME-GAN/train.py)
- 评测/演示入口：[test.py](file:///d:/RME-GAN/test.py)
- 网页演示入口（Streamlit）：[streamlit_app.py](file:///d:/RME-GAN/streamlit_app.py)
- 数据加载（RadioMapSeer 风格）：[loaders.py](file:///d:/RME-GAN/lib/loaders.py)
- 模型结构（RadioWNet）：[modules.py](file:///d:/RME-GAN/lib/modules.py)

## 环境与数据准备

### Python 依赖

本仓库基于 PyTorch、numpy、scikit-image、scipy 等依赖。请按你的环境自行安装（建议使用虚拟环境）。

### 数据目录

当前数据加载器按 RadioMapSeer 的目录组织读取数据。你可以通过两种方式指定数据根目录：

- 方式 1：环境变量 `RME_DATASET_DIR`
- 方式 2：命令行参数 `--dataset_dir`

示例（Windows PowerShell）：

```bash
$env:RME_DATASET_DIR="D:\datasets\RadioMapSeer"
```

## 训练

训练支持 3 种稀疏采样设置（与原始代码一致）：

- setup=1：uniform（固定约 1% 采样）
- setup=2：twoside
- setup=3：nonuniform

示例：

```bash
python train.py --dataset radiomapseer_polygon --dataset_dir "D:\RME-GAN\RadioMapSeer" --setup 1 --epochs 50 --batch_size 30 --exp_index 1
```

训练会在实验目录（如 `radiomapseer_polygon_uniform_1/`）中保存：

- `Trained_ModelMSE_G.pt`：生成器权重（state_dict）
- `Trained_ModelMSE_D.pt`：判别器权重（state_dict）
- `checkpoint_last.pt`：断点续训检查点（含模型/优化器/随机状态）
- `MSE_train.csv`、`MSE_val.csv`：训练/验证曲线

断点续训示例：

```bash
python train.py --dataset radiomapseer_polygon --dataset_dir "D:\RME-GAN\RadioMapSeer" --setup 1 --epochs 50 --batch_size 30 --exp_index 1 --resume auto
```

## 评测与 Demo（输出图片 + 误差统计）

`test.py` 支持 3 种模式：

- `model`：使用训练权重/检查点推理
- `baseline_prior`：不训练，直接用 `prior` 通道做演示
- `baseline_samples`：不训练，直接用稀疏采样通道做演示

示例 1（模型评测）：

```bash
python test.py --dataset radiomapseer_polygon --dataset_dir "D:\RME-GAN\RadioMapSeer" --setup 1 --mode model --weights "radiomapseer_polygon_uniform_1\checkpoint_last.pt" --out_dir "demo_out" --save_images --max_batches 1
```

示例 2（无训练快速 Demo）：

```bash
python test.py --dataset radiomapseer_polygon --dataset_dir "D:\RME-GAN\RadioMapSeer" --setup 1 --mode baseline_prior --out_dir "demo_out" --save_images --max_batches 1
```

默认会输出指标：

- `mae` / `rmse` / `p90` / `p95`：以“评测单位”为准的误差统计（见下方 dB 映射）
- `within_threshold_ratio`：误差 ≤ `--threshold_db` 的像素占比（默认 7 dB）
- `infer_seconds`：本次评测实际推理耗时（不含数据加载/预处理）

输出目录说明（`--save_images` 开启时）：

- `in_buildings_*.png`：输入通道 buildings/env（建筑物/环境栅格）
- `in_tx_*.png`：输入通道 tx_mask（发射点位掩码）
- `in_samples_*.png`：输入通道 sparse_samples（稀疏测量点）
- `in_prior_*.png`：输入通道 prior（先验图）
- `pred_*.png`：预测覆盖图
- `gt_*.png`：真值覆盖图
- `metrics.json`：配置与指标结果

## Streamlit 网页 Demo（推荐答辩展示）

安装并启动：

```bash
pip install streamlit
streamlit run streamlit_app.py
```

网页功能：

- 一键切换 `baseline_prior / baseline_samples / model`
- 显示 MAE/RMSE/P95/≤7dB 占比/推理耗时
- 图像含义在页面直接说明，且标题含具体文件名
- 支持“预测与真值使用彩色热力图”开关（避免黑白图不直观）

### 网页 Demo 演示文档（可直接照着讲）

本节用于快速答辩演示。当前可先复用同一个 `checkpoint_last.pt` 进行展示，不影响流程完整性。

#### 1) 启动网页

```bash
streamlit run streamlit_app.py
```

#### 2) 演示配置（推荐）

- `数据集`：先选 `urbanradio3d`
- `dataset_dir`：`D:\RME-GAN\UrbanRadio3D-main(1)\UrbanRadio3D-main`
- `模式`：先选 `model`
- `weights`：`D:\RME-GAN\UrbanRadio3D-main(1)\UrbanRadio3D-main\checkpoint_last.pt`
- `setup`：`1`
- `max_batches`：`1`（演示更快）
- 勾选：`保存图片到 out_dir`、`预测与真值使用彩色热力图`
- 点击：`运行 Demo`

#### 3) 页面讲解顺序（建议 2-3 分钟）

1. 左侧指标区：先讲 `rmse`、`p95`、`≤7dB占比`、`infer_seconds`。  
2. 右侧样例区：按“输入通道 -> 预测 -> 真值”解释每张图。  
3. 强调彩色热力图：更直观看覆盖强弱变化。  
4. 下载 `metrics.json`：说明结果可复现、可归档。  

#### 4) 输出文件说明（答辩可展示目录）

运行后在 `out_dir`（默认 `demo_out_streamlit`）生成：

- `in_buildings_*.png`：环境/建筑输入
- `in_tx_*.png`：发射点位输入
- `in_samples_*.png`：稀疏测量输入
- `in_prior_*.png`：先验图输入
- `pred_*.png`：预测覆盖图
- `gt_*.png`：真值覆盖图
- `metrics.json`：指标与配置

#### 5) 备用演示方案（无需模型）

如果现场权重加载异常，可切到：

- `模式 = baseline_prior`（最稳）
- 或 `模式 = baseline_samples`

这样无需模型也能完整演示“输入 -> 输出 -> 指标 -> 导出结果”的流程。

### 关于 dB 映射（必须补齐才能对齐“<7 dB”）

本仓库的标签/输出来自 RadioMapSeer 的图像归一化流程（0~256 的连续值），并不直接等同于路径损耗(dB)或 RSRP(dBm)。为了按比赛口径评测，你需要明确“模型输出值 → dB”的映射。

`test.py` 提供两种线性映射方式（选其一）：

- 方式 A：`--value_scale` / `--value_offset`（直接线性变换到 dB）
- 方式 B：`--db_min` / `--db_max`（将值按 `x/256` 映射到 `[db_min, db_max]`）

示例（方式 B）：

```bash
python test.py --dataset_dir "D:\datasets\RadioMapSeer" --setup 3 --weights "uniform_1\Trained_ModelMSE_G.pt" --db_min -140 --db_max -40 --threshold_db 7
```

## 为了满足比赛要求，建议的完善方向

- 数据与场景：引入至少 3 类典型场景的数据（并明确场景标签/划分），实现“分场景训练与分场景评测”输出。
- 指标口径：把输出统一到路径损耗/RSRP 的 dB 单位，提供 MAE/RMSE/P95、≤7 dB 占比等指标，并与实测数据对齐坐标系/天线参数/频段等元信息。
- 工程化推理：补齐从“站点表 + 地图/GIS + 稀疏测量点”到模型输入栅格的生成；实现 1 km² / ≥20 Tx 的批处理推理与计时基准；支持 GPU batch 推理与混合精度。

## 参赛完善准备清单（可直接作为需求文档）

本节用于把“需要准备哪些东西”落到可执行的输入/输出/验收标准上，方便团队分工与对齐口径。

### 1) 数据与标注（必须）

#### 1.1 测量数据表（推荐 CSV/Parquet）

每条测量记录至少包含：

- 位置：`x,y`（米制投影坐标）或 `lon,lat`（WGS84，经纬度）
- 指标：`rsrp_dbm` 或 `pathloss_db`（必须明确单位与定义）
- 关联小区：`serving_cell_id`（或能通过 PCI/ECGI 等字段映射到具体发射点）

建议包含（用于清洗与复现）：

- `time`（时间戳）
- `rsrq_db` / `sinr_db`（若有）
- `meas_source`（路测/网管/仿真等来源）

#### 1.2 发射点/小区参数表（每个 1 km² 场景内 ≥ 20 个）

每个小区至少包含：

- `cell_id`
- 位置：`x,y` 或 `lon,lat`
- `freq_mhz`（或频段标识）
- `tx_power_dbm` 或 `eirp_dbm`（二选一，但必须能闭环到接收功率/路径损耗）
- `antenna_height_m`
- `azimuth_deg`、`tilt_deg`（推荐至少方位角/下倾）

#### 1.3 环境数据（GIS/栅格，至少一种可用）

至少准备建筑物或地形之一：

- 建筑物：建筑轮廓（shp/geojson）或建筑高度栅格（DSM/建筑高度 GeoTIFF）
- 地形：DTM/DEM（GeoTIFF，推荐）

建议准备（用于三类场景区分与泛化）：

- 土地覆盖/地物类型（开阔地/林地/水体等）
- 道路/街区结构（可选）

#### 1.4 场景标签与划分（必须满足 ≥3 类场景）

至少 3 类典型场景，并为每个样本/区域标注 `scene_type`，例如：

- `urban_highrise`（城市高楼密集区）
- `suburban`（郊区）
- `open_field`（开阔田野）

切分规则要求：

- 训练/验证/测试按“区域”切分（避免同一区域不同站点/不同采样点泄漏到训练集）

### 2) 物理量口径与映射（必须，否则无法验收 < 7 dB）

需要明确主评测指标是：

- 路径损耗 Pathloss（dB），或
- RSRP（dBm）

并补齐以下至少一条可复现的映射链路：

- 直接提供 `pathloss_db` 或 `rsrp_dbm` 作为标签（最推荐）
- 若只提供接收功率/RSRP，需明确发射侧口径（`tx_power_dbm`/`eirp_dbm`、带宽/参考信号定义等）以保证不同数据源可对齐

网格化要求：

- 明确栅格分辨率（如 5m/pixel、10m/pixel）
- 明确 1 km² 区域边界定义（bbox 或 polygon）及其坐标系

### 3) 工程化系统输入/输出（必须，用于“系统”交付）

系统输入（建议最小闭环）：

- 区域范围：1 km²（bbox/polygon）
- 区域内站点列表：≥20 发射点
- 区域内稀疏测量点（可选但推荐，体现“由稀疏观测补全覆盖图”）
- 区域环境数据：建筑/地形（用于遮挡与传播差异）

系统输出（建议统一口径）：

- 覆盖图栅格：每站点覆盖图或合成覆盖图（单位：dB/dBm，必须写清楚）
- 指标报告：MAE/RMSE/P90/P95、误差 ≤ 7 dB 的占比
- 运行耗时：至少给出 1 km² / ≥20 站点 / 复杂城市场景的一次端到端耗时

### 4) 评测与验收集（必须）

至少准备 3 个固定验收场景（每类场景至少 1 个；更稳妥是每类 ≥3 个）：

- 每个场景面积约 1 km²
- 每个场景站点数 ≥ 20
- 每个场景有足够实测点覆盖（建议每个站点至少数百到上千有效测量点）

建议固定输出一张汇总表（按场景分组）：

- `scene_type`
- `metric_type`（PL/RSRP）
- `mae_db`、`rmse_db`、`p95_db`
- `within_7db_ratio`
- `runtime_minutes`

### 5) 计算效率（必须满足 ≤30 分钟/1 km²/≥20 站点）

需要写清楚：

- 运行环境：CPU/GPU 型号、显存、内存、PyTorch 版本
- 计时范围：端到端（含栅格化预处理）或纯模型推理（必须说明，并建议两者都给）

### 6) Demo 演示交付（必须）

至少准备 1 个可复现 Demo 场景（可脱敏）：

- 站点表 + 测量表 + 环境数据（或可下载链接）
- 演示内容：输入可视化、预测覆盖图、与实测对比、关键指标、耗时

## 论文引用

如需引用 RME-GAN 相关工作：

```
@ARTICLE{10130091,
  author={Zhang, Songyang and Wijesinghe, Achintha and Ding, Zhi},
  journal={IEEE Internet of Things Journal},
  title={RME-GAN: A Learning Framework for Radio Map Estimation Based on Conditional Generative Adversarial Network},
  year={2023},
  volume={10},
  number={20},
  pages={18016-18027},
  doi={10.1109/JIOT.2023.3278235}}
```
