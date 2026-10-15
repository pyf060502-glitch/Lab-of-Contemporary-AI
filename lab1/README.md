# Lab1--文本分类实验

## 文件与环境

Python 代码共有六个文件如下：

| 文件                      | 职责                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------ |
| `data_preprocessing.py` | 原始带标签数据校验、分层分组划分、确定性清洗                                         |
| `tfidf_features.py`     | 训练原文拟合TF-IDF，保存稀疏矩阵                                                     |
| `train_models.py`       | 利用SVC实现SVM；构建MLP，通过MBGD逐步收敛，进行逐轮监控和梯度自检                    |
| `evaluate_models.py`    | 验证评估，冻结最佳配置                                                               |
| `plot_results.py`       | 生成混淆矩阵图、横向对比图、SVM训练集/测试集Accuracy对比图、MLP的Loss/Accuracy曲线图 |
| `test_predict.py`       | 冻结后全量重训、测试预测及输出预测结果                                               |

在项目lab1文件夹下目录的 PowerShell 运行：

```powershell
conda env create -f environment.yml
conda activate ai-lab1
```

以下命令也在项目lab1文件夹下目录运行，不依赖本机 Miniconda 绝对路径。已运行结果保存在 `outputs`文件夹里。

## 从原始数据复现

原始数据放在 `data-Project1/data-Project1` 文件夹下，其中包含 `train_data.csv` 和 `test_data_unlabeled.csv`。下面按照数据处理、特征提取、模型训练、验证评估、绘图、测试预测的顺序进行复现，新的结果保存在 `reproduction` 文件夹中。

```powershell
# 检查代码中的梯度计算和预测接口
python train_models.py --self-check

# 数据划分、清洗以及TF-IDF特征提取
python data_preprocessing.py --output reproduction
python tfidf_features.py --output reproduction

# 使用默认参数训练SVM和MLP，MLP完整训练150轮
python train_models.py --output reproduction

# SVM调整C值，MLP调整正则化参数
python train_models.py --output reproduction --model svm --svm-c 10 --run-id svc-c10
python train_models.py --output reproduction --model mlp --regularization 0.001 --run-id mlp-reg001

# MLP调整隐藏层神经元数量
python train_models.py --output reproduction --model mlp --hidden 50 --run-id mlp-hidden50
python train_models.py --output reproduction --model mlp --hidden 200 --run-id mlp-hidden200

# 固定200个神经元，改变随机种子查看结果波动
python train_models.py --output reproduction --model mlp --hidden 200 --seed 43 --run-id mlp-hidden200-seed43
python train_models.py --output reproduction --model mlp --hidden 200 --seed 44 --run-id mlp-hidden200-seed44

# 在验证集上评估、选择并冻结配置，然后生成图片
python evaluate_models.py --output reproduction
python plot_results.py --output reproduction

# 利用全部带标签数据重新训练选定模型，并预测测试集
python test_predict.py --output reproduction
```

每一步都使用同一个 `--output reproduction`。如果该目录已有结果，需要换一个新的输出目录；同一个目录中也不能重复使用已有的 `--run-id`。MLP最终重训采用验证集选定的轮数，不一定是150轮。

## 现有输出内容

现有实验结果保存在 `outputs` 文件夹下，具体内容如下：

| 文件夹          | 内容                                                                      |
| --------------- | ------------------------------------------------------------------------- |
| `data/`       | 清洗后的训练集、验证集，以及数据划分信息                                  |
| `features/`   | TF-IDF向量化器、训练集和验证集的稀疏矩阵、标签及特征配置                  |
| `runs/`       | 各次模型训练的参数、模型文件、验证指标及预测；MLP还包含逐轮训练和验证记录 |
| `evaluation/` | 各模型的结果对比表、冻结的最佳配置以及重复种子的结果统计                  |
| `figures/`    | 实验结果图片，共六张，具体如下表                                          |
| `final/`      | 选定模型全量重训后的模型文件、测试集预测结果和输出检查信息                |

| 图片文件                     | 内容                                       |
| ---------------------------- | ------------------------------------------ |
| `svm_confusion_matrix.png` | SVM验证集混淆矩阵                          |
| `svm_accuracy.png`         | SVM训练集和验证集Accuracy对比              |
| `mlp_confusion_matrix.png` | MLP验证集混淆矩阵                          |
| `mlp_loss.png`             | MLP训练集和验证集Loss曲线                  |
| `mlp_accuracy.png`         | MLP训练集和验证集Accuracy曲线              |
| `model_comparison.png`     | SVM和MLP的验证集Accuracy、Macro-F1横向对比 |

混淆矩阵图展现了模型在验证集上预测时的命中/不命中情况，同一模型下的loss/accuracy曲线图展示了模型的收敛/拟合情况，横向对比图展示了不同模型训练指标的差异。

`final/predictions.csv` 是最终测试集预测结果，共2457行，单列整数标签，无表头、无索引，行序与原始测试集一致。测试集没有标签，因此没有测试集Accuracy或Loss。

为了方便查看，最终测试集预测结果 `predictions.csv`和六张图片的figure文件夹在文件夹outputs外各复制了一份。
