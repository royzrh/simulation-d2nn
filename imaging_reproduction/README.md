# 全光成像复现（Lin et al., Science 2018）

所有成像代码和结果在此目录内；不修改已有分类模型。使用
`C:\Users\dell\miniforge3\envs\d2nn\python.exe`。

## 文件整理

根目录只保留 `train.py` 一个训练入口。光学模型、数据读取和训练循环合并在根目录 `train.py`；独立验证及评估工具放入 `imaging/`，每次训练使用独立名称，结果仍写入 `results/<实验名称>/`，不复制模型和验证代码。

| 目录或入口 | 用途 |
| --- | --- |
| `train.py` | 完整训练代码，包含光学模型、数据读取、损失计算、训练、验证和保存，附中文分段注释 |
| `experiments/wavelength1550_visible.ps1` | 1550 nm 五层主实验入口（验证、训练、评估、出图） |
| `experiments/large_sample_visible.ps1` | 10,000 张训练样本对照入口 |
| `experiments/suite1550_visible.ps1` | 层数和层间距系列实验入口 |
| `experiments/history/` | 既有初始化、学习率及粗采样实验入口，保留追溯 |
| `imaging/validation/` | 训练前物理验证、采样覆盖检查 |
| `imaging/datasets/` | 数据获取和大样本数据准备 |
| `imaging/evaluation/` | 测试、数值收敛核查、仿真图、重建、误差分析和报告 |
| `imaging/workflows/` | 实验排队、阶段衔接及验证集选模 |
| `misc/tools/` | 独立评估、绘图和实时状态查看的终端工具 |
| `misc/` | 运行状态、日志、缓存及辅助内容；整理位置记录为 `code_layout.json` |
| `references/` | 论文页面摘录和参考图像 |
| `data/`、`data_imagenet_train10000/` | 原有数据集及其划分 |
| `results/` | 原有模型、配置、指标、仿真图和报告 |

**当前所有训练和队列均已按用户要求停止。整理文件不会恢复训练。** 上述启动脚本保留原有参数（包括原设置的 50 轮）；不要将文件整理理解为批准再次运行这些实验。

`results/图像索引.md` 是已有实验的图像入口。每轮相位图和训练曲线预览可由 `misc/tools/figures_visible.ps1` 单独启动；当前未启动。原始 ImageNet 图像仍在上级 `data/ImageNet/`。已完成的一次性辅助内容保留于 `misc/completed_auxiliary/`。

## 原文对齐

实验室波长已更新为 **1550 nm**。现有已运行实验保留为论文0.4 THz参考；
`misc/queue_control.json` 已阻止队列后续启动不匹配波长的新训练，当前已运行训练不受影响。
1550 nm实验的器件、像元尺寸、有效口径、逐段间距及材料参数须另行确定。
用户已授权先保留论文几何进行数值训练：输入4 mm、层间4 mm、输出7 mm，
像元300 μm、90 mm口径、五层；当前正式训练在 `results/wavelength1550_paper_geometry_os16`。
原 `wavelength1550_paper_geometry50` 因OS2采样误差停止，仅保留为诊断。
正式训练采用OS16和已核查的较小空白边界；逐张累积梯度，实现每4张图一次Adam更新。
正式评估还用独立CPU传播实现对固定验证图比较OS8/16/32，保存原始输出和误差图。
完整主模型/大样本评估还增加OS48有限参考；传播函数按128行重算，避免缓存大尺寸完整传递数组。
既有匹配权重的采样核查可通过`imaging/evaluation/evaluate.py --checkpoint <权重路径> --device cpu --extended-resolution-audit-only`扩展。
扩展结果复用`evaluation/high_resolution_validation`的JSON、NPZ和图像，不生成另一套结果目录。
该检查固定同一组相位，不增加训练，也不把有限参考或两张图的检查称为完整收敛证明。
后续层数/间距扫描通过 `imaging/workflows/run_suite.py --stage adapted1550` 串行执行，
等待当前主模型及10,000张训练对照全部完成；每项保存独立采样核查和综合图。
层数扫描为0/1/2/3/4/5/10层、总距离30 mm；五层间距扫描为1～10 mm。
输入始终4 mm，单层输出26 mm，其余输出7 mm；未知分段分配仍明确标为假设。
1550 nm扫描曲线与0.4 THz参考曲线分别绘制，避免混用不同波长的数据。
该几何为当前仿真假设，未确认为实验室器件尺寸；其余训练设置详见 `paper_alignment.md`。
`train.py` 支持显式 `--wavelength-nm` 和 `--pitch-um`，不会自动套用太赫兹材料参数。
当前 `train.py` 的吸收/高度误差分支仍采用原文太赫兹材料常数，不能用于1550 nm硬件验收。

详见 `paper_alignment.md`。原始两份 PDF 位于上级目录。

主模型使用 0.4 THz、5 层、每层 300×300、0.3 mm 像元（90 mm 边长）、
输入 4 mm、四段层间距各 4 mm、输出 7 mm、相位 0～2π。
ImageNet 随机子集 2000 张，按 1500/200/300 划分；50 epochs、batch=4、Adam。
原作者具体图像索引、学习率、相位初始化、灰度幅度/强度约定、SSIM 设置、FFT
网格和重建软件设置没有完整提供；必须保留这些选择及敏感性记录，不能声称逐数值严格相同。

当前默认将灰度作为输入幅度，目标强度为灰度平方，直接最小化输出强度 MSE。
另提供 `--target gray_intensity` 作为定义敏感性对照；不能根据测试集选定义。
不在前向或损失中拟合每张图的增益，不把归一化后的图像效果当作光学效率。
输出同时报告强度 SSIM、幅度 SSIM、原始 MSE 与收集功率比例。

## 运行

```powershell
$py = 'C:\Users\dell\miniforge3\envs\d2nn\python.exe'
cd H:\PROJECT\study\simulation_d2nn\imaging_reproduction
& $py imaging/validation/check_physics.py
& $py imaging/datasets/acquire_imagenet.py
& $py train.py --name baseline_os2 --oversample 2
```

训练命令不会自动访问测试集；最终验收脚本在固定配置后单独评估。
没有结果文件时不得把“脚本存在”写成实验完成。

## 保留原实验的续训

`--resume` 用于恢复同一目录内中断的实验，要求配置一致。
`--continue-from` 用于将较短实验延长到新的总轮数，并写入新目录；它继承模型、
Adam 状态、历史和最佳验证模型，同时记录父检查点路径与 SHA256。
例如从 10 轮继续到总共 50 轮，会再训练 40 轮，原目录保留。
除总轮数外，模型、数据划分、学习率与训练配置必须相同；目标目录必须为空。

每次训练（包括短验证训练）均通过可见终端启动，实时显示进度。
每项正式实验的终端入口集中于 `experiments/`，独立评估工具位于 `misc/tools/`；训练结果保存在 `results/`。

## 双方向分辨率图

`resolution_chart_both_axes` 为六档线宽分别提供水平、竖直条纹，最细为 1.2 mm。
它只用于模型评估，不加入 ImageNet 训练数据。沿条纹长度平均后的剖面用于
计算调制度，以降低单个散斑像素的影响。条纹位置和剖面平均范围属于本项目
明确选定的评估设置，不声称与作者未公开的测试图逐像素一致。
旧版单方向图仍保留；单独重评的新结果写入
`synthetic_metrics_selected_patterns.json`，不能与旧版测量方法混用。

已合并的旧光学模型、数据读取和训练文件集中保留于 `misc/completed_auxiliary/merged_training_sources/`，仅供追溯，不参与运行；删除被自动审批策略拦截。

新训练不传 `--wavelength-nm` 时默认使用 1550 nm；恢复训练时继承原检查点波长。与项目波长控制设置冲突会直接报错退出，不再无限等待。`Geometry` 的默认频率仍保留论文参考值，供旧模型评估使用。
