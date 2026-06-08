# 项目详细说明：四位数字验证码识别与智能评估系统

## 1. 项目概述

本项目是在原始“单个手写数字识别”项目基础上重新开发的新系统。原项目主要由 `app.py`、`CNN_Model_Training.ipynb` 和 `cnn_model_v2.h5` 组成，功能集中在 MNIST 单数字识别和 Gradio 画板推理。

新项目将目标扩展为四位数字验证码识别，并增加数据集管理、真实数据构建、模型训练评估、大模型辅助质检、大模型错误解释和训练报告生成能力。

所有新代码和文件都放在 `captcha_recognition_ai/` 目录中，避免污染原始项目文件。

## 2. 相比原项目的主要改进

### 2.1 识别任务升级

原项目识别单个数字，新项目识别四位数字验证码。

原项目输入通常是用户在画板中绘制的单个数字，新项目输入是一张四位数字图片，并要求模型输出完整四位数字序列。

### 2.2 数据流程升级

原项目主要使用 MNIST 训练单数字模型。新项目增加完整数据流水线：

- 真实单数字图片准备。
- 四位验证码图片构建。
- 轻度模糊、扭曲、噪声和干扰线增强。
- 标签文件保存。
- 训练集、验证集、测试集划分。
- 数据质量统计和样本标记。

### 2.3 模型结构升级

新项目提供四输出头 CNN：

- 共享卷积特征提取层。
- 四个输出头分别预测第 1、2、3、4 位数字。
- 每个输出头都是 10 分类，对应数字 0 到 9。

这种结构适合固定四位验证码，训练和调试都比较直观。

### 2.4 评估体系升级

新项目不仅记录单字符准确率，还记录整张验证码是否完全识别正确。

评估输出包括：

- `metrics.json`
- `wrong_predictions.csv`
- `confusion_matrix.png`
- `error_analysis.md`
- `training_report.md`

### 2.5 大模型辅助能力

新项目增加三个可选大模型模块：

- 多模态大模型辅助数据质量审核。
- 多模态大模型辅助解释错误样本。
- 文本大模型自动生成训练报告。

这些模块只做辅助分析，不生成训练图片，不替代真实标签。

## 3. 项目目录结构

```text
captcha_recognition_ai/
├─ app.py
├─ README.md
├─ requirements.txt
├─ .env.example
├─ data/
│  ├─ raw_digits/
│  ├─ captcha_images/
│  ├─ raw_digits.csv
│  ├─ captcha_labels.csv
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ docs/
│  ├─ PROJECT_DETAILS.md
│  └─ DATA_SOURCES.md
├─ models/
│  └─ captcha_cnn.keras
├─ reports/
│  ├─ metrics.json
│  ├─ wrong_predictions.csv
│  ├─ confusion_matrix.png
│  ├─ error_analysis.md
│  └─ training_report.md
└─ src/
   ├─ config.py
   ├─ data/
   │  ├─ prepare_digits.py
   │  ├─ download_dataset.py
   │  ├─ augment.py
   │  ├─ build_captcha_dataset.py
   │  ├─ dataset_manager.py
   │  └─ quality_check.py
   ├─ models/
   │  ├─ single_digit_cnn.py
   │  ├─ captcha_cnn.py
   │  └─ crnn.py
   ├─ training/
   │  ├─ data_loader.py
   │  ├─ train_captcha.py
   │  ├─ train_single_digit.py
   │  └─ evaluate.py
   ├─ llm/
   │  ├─ client.py
   │  ├─ vision_quality_review.py
   │  ├─ error_explainer.py
   │  └─ report_generator.py
   └─ ui/
      ├─ login_demo.py
      └─ data_viewer.py
```

## 4. 数据构建流程

### 4.1 准备真实数字图片

命令：

```bash
python -m src.data.prepare_digits --source sklearn
```

该命令会：

- 读取 `scikit-learn` 的真实手写数字数据集。
- 将每个数字保存为 PNG 图片。
- 生成 `data/raw_digits.csv`。

### 4.2 构建四位验证码数据集

命令：

```bash
python -m src.data.build_captcha_dataset --count 5000
```

该命令会：

- 从真实单数字图片中随机选择 4 个数字。
- 拼接为一张四位验证码图片。
- 应用轻度增强。
- 保存图片到 `data/captcha_images/`。
- 生成 `data/captcha_labels.csv`。
- 自动划分 `train.csv`、`val.csv`、`test.csv`。

### 4.3 数据质量管理

查看统计：

```bash
python -m src.data.dataset_manager stats
```

生成数据质量报告：

```bash
python -m src.data.dataset_manager report
```

本地质量检测：

```bash
python -m src.data.quality_check
```

标记样本质量：

```bash
python -m src.data.dataset_manager mark captcha_000001 too_blurry --note "数字 3 不清晰"
```

## 5. 模型训练流程

训练四位验证码模型：

```bash
python -m src.training.train_captcha --epochs 10 --batch-size 64
```

训练后输出：

- `models/captcha_cnn.keras`
- `reports/training_history.json`

模型结构为四输出头 CNN，每个输出头预测验证码中的一个位置。

## 6. 模型评估流程

命令：

```bash
python -m src.training.evaluate --model models/captcha_cnn.keras
```

评估输出：

- `reports/metrics.json`
- `reports/wrong_predictions.csv`
- `reports/confusion_matrix.png`

核心指标：

- 单字符准确率。
- 第 1、2、3、4 位分别的准确率。
- 四位验证码整体准确率。
- 推理耗时。

## 7. 大模型辅助模块

### 7.1 数据质量审核

命令：

```bash
python -m src.llm.vision_quality_review --sample-size 20
```

输出：

- `data/quality_reviews.csv`
- `reports/llm_quality_review.json`

### 7.2 错误样本解释

命令：

```bash
python -m src.llm.error_explainer --limit 20
```

输出：

- `reports/llm_error_explanations.json`
- `reports/error_analysis.md`

### 7.3 自动训练报告

命令：

```bash
python -m src.llm.report_generator
```

输出：

- `reports/training_report.md`

## 8. 用户界面

启动验证码验证界面：

```bash
python app.py
```

`app.py` 默认启动纯 Python 标准库实现的轻量版验证码验证界面，不加载 TensorFlow 模型，也不依赖 Gradio 前端资源。

如需启动 Gradio 版本：

```bash
python -c "from src.ui.login_demo import demo; demo.launch()"
```

界面包含：

- 验证码图片。
- 四位数字输入框。
- 提交验证按钮。
- 刷新验证码按钮。
- 验证结果提示。

系统会直接比较用户输入与验证码真实标签，模拟登录验证码校验流程。

轻量版网页提供 `/agent` 页面，用于解释 Agent 在项目中的作用。需要注意，Agent 不参与前端验证码作答，而是在数据审核、错误分析和训练报告生成阶段辅助开发者理解模型和数据。

## 9. 学习重点

通过本项目可以学习：

- 如何把单数字识别扩展为多字符验证码识别。
- 如何设计真实数据驱动的机器学习流水线。
- 如何管理标签、划分数据集和避免前导零丢失。
- 如何设计四输出头 CNN。
- 如何记录整体验证码准确率。
- 如何分析错误样本。
- 如何把大模型作为辅助评估工具接入机器学习项目。

## 10. 后续可继续扩展

后续可以进一步加入：

- 更多真实数字数据集。
- 人工复核 Web 页面。
- 主动学习：优先复查模型最不确定样本。
- CRNN 或 Transformer 序列识别模型。
- 在线收集用户验证码识别反馈。
- 模型版本管理。
- 实验配置管理。
- 更完整的登录模拟系统。
