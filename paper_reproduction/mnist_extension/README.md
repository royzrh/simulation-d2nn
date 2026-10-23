# MNIST 扩展复现：Lin et al. (Science, 2018), Fig. S1 / S2

**用途：** 在你已有的 `paper_reproduction` 工程上准备 MNIST 扩展实验。所有新文件均位于独立的 `mnist_extension` 子目录中，不需要修改 `step7_4_full_gpu_training.py` 或 `fashion_mnist_phase_train.py`。

## 1. 目录安装

解压本 ZIP 后，把其中的 `mnist_extension` 整个文件夹放到：

```text
H:\PROJECT\study\simulation_d2nn\paper_reproduction\mnist_extension\
```

推荐在 PowerShell 中进入项目目录：

```powershell
cd H:\PROJECT\study\simulation_d2nn
conda activate d2nn
```

## 2. 当前 GPU 繁忙时：只做 CPU 检验

```powershell
python -m compileall -q .\paper_reproduction\mnist_extension
python .\paper_reproduction\mnist_extension\check_cpu.py
```

期待显示：

```text
[PASS] 1-layer phase: ...
[PASS] 1-layer complex: ...
[PASS] 5-layer complex: ...
[PASS] 7-layer patch: old 5 layers unchanged, added layers trainable
ALL CPU CHECKS PASSED (no GPU used)
```

`check_cpu.py` 使用**随机图像验证光学传播、梯度、冻结机制**，并不训练真实 MNIST，也不代表真实分类准确率。为了避免影响当前 GPU 任务，不要在 GPU 繁忙时运行下面的训练命令。

## 3. 等当前两个 GPU 任务结束后：Fig. S1

先用以下命令做一次小规模 GPU 烟雾测试：

```powershell
python -u .\paper_reproduction\mnist_extension\train_extension.py --stage smoke --layers 1 --modulation phase --device cuda --oversample 4 --batch-size 8 --epochs 1
```

然后依次完成需要的正式扩展实验（推荐一次只运行一个）：

**单层相位型：**

```powershell
python -u .\paper_reproduction\mnist_extension\train_extension.py --stage full --layers 1 --modulation phase --device cuda --oversample 4 --batch-size 8 --epochs 10 --input-gap-mm 30 --detector-side-mm 4.8
```

**单层复振幅型：**

```powershell
python -u .\paper_reproduction\mnist_extension\train_extension.py --stage full --layers 1 --modulation complex --device cuda --oversample 4 --batch-size 8 --epochs 10 --input-gap-mm 30 --detector-side-mm 4.8
```

**五层复振幅型（也是 Fig. S2 的基础网络）：**

```powershell
python -u .\paper_reproduction\mnist_extension\train_extension.py --stage full --layers 5 --modulation complex --device cuda --oversample 4 --batch-size 8 --epochs 10 --input-gap-mm 30 --detector-side-mm 4.8
```

你已经在 Step 7.4 中训练了**五层相位型**；只要其几何、探测器、损失和数据划分与本次扩展配置一致，就可以将已有结果作为对照，不必重复训练。

## 4. Fig. S2：五层 COMPLEX 基础网络 + 两层 Patch

此实验使用 Fig. S1 **五层复振幅型**的 `best_model.pt` 作为基础模型；程序明确拒绝把五层相位型权重当作论文的复振幅基础网络。

```powershell
$base = 'H:\PROJECT\study\simulation_d2nn\paper_reproduction\mnist_extension\results\figS1\在此填写五层complex结果文件夹\best_model.pt'

python -u .\paper_reproduction\mnist_extension\train_patch.py --base-checkpoint $base --stage full --device cuda --oversample 4 --batch-size 8 --epochs 10 --input-gap-mm 30 --detector-side-mm 0.8
```

Patch 程序载入五层基础相位和振幅，**严格检查优化更新前后原五层权重保持不变**；新增 L6–L7 可训练。采用 30 mm 间距是假设值，原论文 Patch 的精确放置仍需进一步核实。

## 5. 模型文件与输出

各次运行有独立的时间戳结果目录，输出：

- `best_model.pt`：根据**验证集**选出的最佳模型。
- `last_checkpoint.pt`：完整状态，可在中断后继续。
- `config.json`：保存几何、训练参数、数据划分 ID。
- `training_history.csv`、`training_curves.png`：训练过程。
- `confusion_matrix.png`、`phase_layers.png`：分类与相位图。
- `amplitude_layers.png`：复振幅模型的振幅掩模图。
- `test_metrics.json`：**仅 full** 阶段，对冻结的最佳模型计算官方 MNIST 测试集结果。
- `validation_metrics.json`：smoke/pilot 仅验证，不访问官方测试集。

训练意外中断后，用相同的参数并增加：

```powershell
--resume '完整路径\last_checkpoint.pt'
```

注意：其他所有选项必须与原任务一致（包括 `--epochs`），避免错误续训。

## 6. 独立验证

```powershell
python .\paper_reproduction\mnist_extension\evaluate_extension.py --checkpoint '完整路径\best_model.pt' --dataset validation --device cuda
```

只有所有结构及几何超参数都选好后，才对 full 检查点使用 `--dataset test`，避免用测试集选择模型。

## 7. 论文匹配与局限

| 原论文图 | 模型 | 原论文 MNIST 测试准确率 | 本项目状态 |
|---|---|---:|---|
| Fig. S1 | N=1 phase-only | 55.64% | 可复现；采用相同像素间距 |
| Fig. S1 | N=1 complex-valued | 64.84% | 可复现；复振幅具体参数化仍有差异 |
| Fig. S1 | N=5 phase-only | 91.75% | 已有 Step 7.4 对照 |
| Fig. S1 | N=5 complex-valued | 93.23% | 可复现；作为 S2 的基础模型 |
| Fig. S2 | N=5 complex + 2-layer patch | 93.39% | 可复现设计思路；准确几何未完全确认 |

**非常重要：** 上述为原论文报告的数字，**不是本项目的预测结果**。复现方案仍然存在以下不可混淆的假设：

- MNIST 二值化阈值为 0.5，输入物理宽度按 `28 × 0.4mm × input-scale` 定义；确切论文输入宽度未得到证实。
- 单层实验的样品到层、层到输出 30 mm 与 Fig. S1 文字匹配；五层分类器第一层输入距离采用 30 mm 作为候选假设，并未得到独立验证。
- 原工程 Step 7.4 的探测器中心坐标为暂定值；4.8 mm 的方形检测区来自 Fig. S1；Patch 默认 0.8 mm 来自 Fig. S2。
- 复振幅定义 `sigmoid(raw_amplitude)*exp(i*pi*sigmoid(raw_phase))` 是本项目选定的可微实现，不声称与原 TensorFlow 模型的参数化和初始化严格一致。
- `baseline` / `joint` 损失沿用 Step 7.4，是自定义目标函数，未确认与原文精确相同。
- ASM 使用 OS=4 和 pad=2，是数值离散设置，论文未规定相同的 FFT 实现。
- Fig. S2 的新增层间距暂按 30 mm；原论文中“patched right at the exit”需进一步按图确定精确物理位置。

原始文献：Xing Lin et al. (2018), *All-optical machine learning using diffractive deep neural networks*, **Science** 361, 1004–1008, DOI: 10.1126/science.aat8084。对照补充材料 Fig. S1（第 21 页）、Fig. S2（第 22–23 页）及 Methods。
