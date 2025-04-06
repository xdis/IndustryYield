# IndustryYield - 工业生产收率预测系统

## 项目英文名称

**IndustryYield**：这个名称简洁明了，反映了项目的核心目的——预测工业生产过程中的产品收率。

## requirements.txt

```
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## README.md

```markdown
# IndustryYield

基于机器学习的工业生产收率预测系统，专注于异烟酸生产过程的收率优化。

## 项目简介

IndustryYield 是一个利用机器学习技术预测工业生产过程中产品收率的系统。本项目以异烟酸生产过程为例，通过分析生产过程中的各项参数（如原料配比、温度、压强、时间等），构建预测模型，帮助生产管理人员提前了解可能的收率情况，优化生产决策。

## 主要特点

- 全面的数据预处理流程，处理缺失值和异常值
- 深入的特征工程，提取多维度特征
- 基于XGBoost的高精度预测模型
- 交叉验证确保模型稳定性和泛化能力
- 可视化分析助力生产过程理解

## 安装方法

1. 克隆仓库到本地：

```bash
git clone https://github.com/yourusername/IndustryYield.git
cd IndustryYield
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

请将以下数据文件放置在data目录下：
- `jinnan_round1_train_20181227.csv`：训练数据集
- `jinnan_round1_testA_20181227.csv`：测试集A
- `jinnan_round1_testB_20190121.csv`：测试集B

可以从[天池竞赛官网](https://tianchi.aliyun.com/competition/entrance/231700/information)下载这些数据文件。

## 使用方法

运行主程序进行预测：

```bash
python 工业生产预测.py
```

程序将自动进行以下步骤：
1. 数据加载与预处理
2. 特征工程与转换
3. 模型训练与交叉验证
4. 测试集预测
5. 结果保存

预测结果将保存在`results`目录下。

## 模型配置

在`工业生产预测.py`中，您可以调整以下参数以优化模型性能：

```python
params_xgb = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'silent': True
}

fit_params = {
    'num_boost_round': 10000,
    'early_stopping_rounds': 200,
    'verbose_eval': 50
}
```

## 项目结构

```
IndustryYield/
├── data/                # 数据文件目录
├── results/             # 预测结果输出目录
├── 工业生产预测.py       # 主程序
├── requirements.txt     # 项目依赖
└── README.md            # 项目说明
```

## 贡献指南

欢迎提交问题和改进建议！请先fork本仓库，然后提交pull request。

## 许可证

MIT

## 联系方式

如有任何问题，请通过[Issues](https://github.com/yourusername/IndustryYield/issues)联系。
```

这个README和requirements.txt文件应该能够帮助其他用户理解项目并成功运行代码。你可以根据自己的实际情况调整GitHub用户名和其他细节。这个README和requirements.txt文件应该能够帮助其他用户理解项目并成功运行代码。你可以根据自己的实际情况调整GitHub用户名和其他细节。