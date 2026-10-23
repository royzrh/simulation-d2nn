# 1550 nm 五层 D2NN 固定总长距离实验

仅改变四段层间距 `z1, z2, z3, z4`（mm），要求全部为正且和为 400 mm。输入传播固定 30 mm，输出传播固定 100 mm，全光路 530 mm。五层均为 200×200，像素间距 50 μm，孔径 10 mm，相位保持原模型的 `π·sigmoid` 约束。

## 环境和运行位置

最终目录：`H:\PROJECT\study\simulation_d2nn\1550\test`。以下命令均在该目录执行：

```powershell
Set-Location 'H:\PROJECT\study\simulation_d2nn\1550\test'
```

此机可用解释器为 `C:\Users\dell\miniforge3\envs\d2nn\python.exe`，运行时可将下列命令中的 `python` 替换为 `& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe'`。依赖 PyTorch、torchvision、NumPy、Matplotlib；生成报告另需 ReportLab，程序会尝试使用本机 Codex 已安装的文档依赖。默认使用 CUDA。

## 一次完成

```powershell
python -B d2nn_spacing.py all --train-count 1000 --val-count 1000 --search-epochs 5 --epochs 15 --batch-size 8 --lr 0.005 --seed 42 --oversample 4 --pad-factor 2
```

流程为九组各 5 轮筛选 → 选验证最佳的非等间距 → 非等间距与等间距均训练到 15 轮 → 各自选择验证最佳检查点 → 官方 10000 张测试 → 图表及中文 PDF。测试集不参与距离或检查点选择。

## 分步执行

```powershell
python -B d2nn_spacing.py search --search-epochs 5
python -B d2nn_spacing.py refine --epochs 15
python -B d2nn_spacing.py evaluate
python -B d2nn_spacing.py reference
python -B d2nn_spacing.py plot
python -B d2nn_spacing.py report
```

单组训练支持独立修改四个距离参数，单位 mm：

```powershell
python -B d2nn_spacing.py train --z1 50 --z2 75 --z3 125 --z4 150 --epochs 15
```

也可使用 `--z 50 75 125 150`；独立参数会覆盖 `--z` 的对应值。修改实验数据量、种子、采样或优化设置时，使用独立输出根目录，例如 `--root results_other`，并在各阶段保持参数一致，避免混合实验。

## 继续训练

训练自动检测对应候选的最后检查点，保留相位、Adam 状态和已完成轮数，从下一轮继续。`--epochs` 表示总目标轮数，不是额外增加的轮数。短程筛选后的 15 轮训练接续第 5 轮最后状态；验证最佳检查点另外保存，供最终评价使用。重复执行已经完成的阶段不会从头重训。

## 公平比较的范围

默认训练子集 1000 张、验证子集 1000 张、seed 42。各组共享固定划分、初相位及每轮数据顺序。本次是独立的小样本受控实验；原有 3000 张训练样本、15 轮实验的数据预算不同，不能将两次结果的差异直接归因于层间距。单种子和有限预设搜索不支持普遍优越性或全局最优结论。

正确探测效率为真实类别区域积分强度除以输入积分强度；总探测捕获率为全部 10 个区域积分强度之和除以输入积分强度。每张输入归一化为 1。这些是标量模拟的积分强度比例，不能当作实际器件经标定的绝对光功率效率。

您的最佳参数完整用于本轮等间距基线（四段各 100 mm）。此外，`reference/best_model.pt` 是您原有最佳模型的本地副本；`reference` 命令用本轮相同的官方 10000 张测试图像复测。该模型训练于 3000 张图像，只作独立参考，不参与本轮 1000 张训练预算下的候选排序或配对比较。存在此副本时，`all` 也会执行这项复测。

## 输出索引

- `results/search_results.json`、`.csv`：九组短程结果和所选非等间距。
- `results/refine_results.json`：两组延长训练及验证最佳检查点信息。
- `results/final_results.json`、`.csv`：官方测试指标及配对预测比较。
- `results/runs/<候选名>/config.json`、`history.json`：配置和逐轮训练/验证历史。
- `results/runs/<候选名>/best_model.pt`、最后检查点：验证选择和训练恢复。
- `results/runs/<候选名>/test_predictions.csv`、`test_metrics.json`：逐样本预测和混淆矩阵。
- `results/search_comparison.png`、`final_comparison.png`、`confusion_*.png`：黑白比较图。
- `results/spacing_metrics.png`：按候选对齐四段距离、准确率及两种探测效率。
- `results/test_comparison.png`：两个验证集所选模型的官方测试指标。
- `results/d2nn_spacing_report.pdf`：基于本次实测结果的中文报告。
- `results/split.json`、`manifest.json`、`data_manifest.json`：固定数据划分、运行环境与本地数据副本的校验记录。
- `results/historical_reference.json`、`reference_evaluation/`：原最佳模型的历史记录、本轮复测及逐样本预测。

所有运行输出均留在本实验目录内。主实现只有 `d2nn_spacing.py`，报告生成函数也包含在该文件中。

## 数值与恢复检查

```powershell
python -B test_d2nn_spacing.py
```

该独立测试文件核对新旧模型的等间距输出、损失和相位梯度，检查 OS4 的 GPU 前后向、自由传播能量守恒，并使用独立合成数据验证中断续训与连续训练的参数完全一致。它不改变正式实验的训练预算或候选选择。


## 本次实测结果

| 模型 | 训练图像 | 测试准确率 | 正确探测效率 | 总捕获率 |
|---|---:|---:|---:|---:|
| 本轮等间距 | 1000 | 86.21% | 0.4809% | 1.0795% |
| 本轮非等间距 | 1000 | 86.69% | 0.5285% | 1.2505% |
| 原最佳模型复测 | 3000 | 88.26% | 0.3754% | 0.7278% |

本轮非等间距为 150 / 125 / 75 / 50 mm，四段合计 400 mm。相同 1000 张训练图像、15 轮预算下，测试准确率提高 0.48 个百分点；这是单种子、小样本结果。原最佳模型训练于 3000 张图像，只作参考。完整非等间距参数见 results/nonuniform_parameters.json，最终一致性检查见 results/result_audit.json。
