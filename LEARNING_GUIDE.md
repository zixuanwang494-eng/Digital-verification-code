# 项目学习导览：从零理解四位数字验证码识别系统

这份文档面向完全不了解本项目的人，用来说明应该如何一步步理解项目目标、目录结构、数据流程、模型训练、验证码界面和 Agent 辅助模块。

建议按本文顺序阅读和运行，不要一开始就直接看所有代码。

## 1. 先理解项目要解决什么问题

本项目的目标是做一个四位数字验证码识别与验证系统。

用户看到一张四位数字验证码图片，例如 `5831`，然后输入自己看到的数字。系统判断用户输入是否和验证码真实标签一致。

和普通验证码页面不同，本项目还包含机器学习训练流程：

- 准备真实数字图片数据。
- 把真实数字图片组合成四位验证码。
- 对验证码做轻度模糊、扭曲、噪声和干扰线处理。
- 训练模型识别验证码中的四个数字。
- 评估模型表现。
- 用 Agent 辅助检查低质量图片、解释错误样本、生成训练报告。

核心原则：

- 训练数据必须来自真实数字图片。
- 不能用 AI 生成图片冒充训练数据。
- Agent 只能辅助分析，不能生成训练数据，也不能替代真实标签。

## 2. 推荐阅读顺序

第一次学习项目时，建议按下面顺序看文档：

1. `README.md`

   了解项目怎么启动、主要命令是什么。

2. `PROJECT_GOALS.md`

   了解项目目标、数据原则和 Agent 辅助边界。

3. `docs/DATA_SOURCES.md`

   了解当前使用的数据集来自哪里，为什么算真实数字图片。

4. `docs/PROJECT_DETAILS.md`

   系统性了解项目结构、模块分工和完整流程。

5. `docs/LEARNING_GUIDE.md`

   也就是本文，用来串联所有内容。

## 3. 先看项目目录结构

项目主目录是：

```text
captcha_recognition_ai/
```

重要文件和目录如下：

```text
captcha_recognition_ai/
├─ app.py
├─ README.md
├─ PROJECT_GOALS.md
├─ requirements.txt
├─ data/
├─ docs/
├─ models/
├─ reports/
└─ src/
```

各部分含义：

- `app.py`：项目默认启动入口，启动轻量验证码网页。
- `README.md`：快速启动说明。
- `PROJECT_GOALS.md`：项目目标基线。
- `requirements.txt`：Python 依赖。
- `data/`：真实数字图片、验证码图片和标签文件。
- `docs/`：项目说明文档。
- `models/`：训练好的模型保存位置。
- `reports/`：评估结果、错误分析、训练报告保存位置。
- `src/`：项目源代码。

## 4. 理解数据目录

数据目录是：

```text
captcha_recognition_ai/data/
```

常见文件：

```text
data/
├─ raw_digits/
├─ captcha_images/
├─ raw_digits.csv
├─ captcha_labels.csv
├─ train.csv
├─ val.csv
└─ test.csv
```

含义：

- `raw_digits/`：真实单数字图片。
- `captcha_images/`：由真实数字图片组合得到的四位验证码图片。
- `raw_digits.csv`：单数字图片的标签和来源记录。
- `captcha_labels.csv`：所有验证码图片的标签和增强参数记录。
- `train.csv`：训练集。
- `val.csv`：验证集。
- `test.csv`：测试集。

学习重点：

- `label` 字段是验证码真实答案。
- 标签必须当字符串处理，因为验证码可能是 `0123`，不能丢掉前导零。
- `augment_params` 记录了每张验证码图片的增强参数，方便复现实验和分析错误。

## 5. 理解源代码目录

源代码目录是：

```text
captcha_recognition_ai/src/
```

主要模块：

```text
src/
├─ config.py
├─ data/
├─ models/
├─ training/
├─ llm/
└─ ui/
```

### 5.1 `src/config.py`

集中管理路径和项目常量。

例如：

- 数据目录。
- 模型目录。
- 报告目录。
- 验证码图片尺寸。
- 验证码长度。
- 数字类别数量。

学习项目时可以先看这个文件，因为很多脚本都依赖它。

### 5.2 `src/data/`

负责数据准备、验证码构建和数据管理。

重要文件：

- `prepare_digits.py`：准备真实单数字图片。
- `build_captcha_dataset.py`：构建四位验证码图片。
- `augment.py`：图像增强，包括模糊、旋转、噪声、干扰线等。
- `dataset_manager.py`：统计数据、标记样本质量。
- `quality_check.py`：本地自动质量检查。

建议重点看：

1. `prepare_digits.py`
2. `build_captcha_dataset.py`
3. `augment.py`

这三个文件能解释验证码图片是如何从真实数字图片构建出来的。

### 5.3 `src/models/`

负责模型结构。

重要文件：

- `captcha_cnn.py`：四输出头 CNN 模型。
- `single_digit_cnn.py`：单数字 CNN，用于保留原项目思路和后续扩展。
- `crnn.py`：未来序列模型扩展入口。

当前核心模型是 `captcha_cnn.py`。

它的思路是：

- 输入一张四位验证码图片。
- 共享 CNN 特征提取层。
- 输出四个结果。
- 每个结果预测对应位置的数字。

例如一张验证码是 `5831`：

- 第一个输出头预测 `5`
- 第二个输出头预测 `8`
- 第三个输出头预测 `3`
- 第四个输出头预测 `1`

### 5.4 `src/training/`

负责训练和评估。

重要文件：

- `data_loader.py`：读取验证码图片和标签。
- `train_captcha.py`：训练四位验证码模型。
- `evaluate.py`：评估模型，生成指标和错误样本。
- `train_single_digit.py`：单数字模型扩展入口。

学习重点：

- `data_loader.py` 如何把 `5831` 拆成四个标签。
- `train_captcha.py` 如何训练四输出模型。
- `evaluate.py` 如何计算整体验证码准确率。

### 5.5 `src/llm/`

负责 Agent 辅助模块。

重要文件：

- `client.py`：大模型调用封装。
- `vision_quality_review.py`：多模态数据质量审核 Agent。
- `error_explainer.py`：错误样本解释 Agent。
- `report_generator.py`：训练报告生成 Agent。

Agent 在本项目中的作用：

- 帮助开发者看图片质量。
- 帮助解释模型为什么识别错。
- 帮助整理训练报告。

Agent 不做的事：

- 不生成训练图片。
- 不替代真实标签。
- 不在验证码页面里自动给用户答案。

### 5.6 `src/ui/`

负责用户界面。

重要文件：

- `simple_server.py`：默认轻量网页，不依赖 Gradio。
- `login_demo.py`：Gradio 版本界面。
- `data_viewer.py`：数据浏览界面扩展。

当前默认启动的是轻量页面：

```bash
python app.py
```

## 6. 完整运行流程

第一次运行项目时，建议按下面顺序执行。

### 6.1 安装依赖

```bash
cd captcha_recognition_ai
pip install -r requirements.txt
```

### 6.2 准备真实数字图片

```bash
python -m src.data.prepare_digits --source sklearn
```

运行后会生成：

- `data/raw_digits/`
- `data/raw_digits.csv`

### 6.3 构建四位验证码数据集

```bash
python -m src.data.build_captcha_dataset --count 5000
```

运行后会生成：

- `data/captcha_images/`
- `data/captcha_labels.csv`
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

### 6.4 查看数据统计

```bash
python -m src.data.dataset_manager stats
```

也可以生成报告：

```bash
python -m src.data.dataset_manager report
```

### 6.5 启动验证码网页

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

Agent 说明页：

```text
http://127.0.0.1:7860/agent
```

### 6.6 训练模型

```bash
python -m src.training.train_captcha --epochs 10
```

训练后会生成：

- `models/captcha_cnn.keras`
- `reports/training_history.json`

### 6.7 评估模型

```bash
python -m src.training.evaluate --model models/captcha_cnn.keras
```

评估后会生成：

- `reports/metrics.json`
- `reports/wrong_predictions.csv`
- `reports/confusion_matrix.png`

### 6.8 运行 Agent 辅助模块

数据质量审核：

```bash
python -m src.llm.vision_quality_review --sample-size 20
```

错误样本解释：

```bash
python -m src.llm.error_explainer --limit 20
```

训练报告生成：

```bash
python -m src.llm.report_generator
```

## 7. 如何理解验证码验证页面

验证码页面的逻辑非常简单：

1. 从 `data/captcha_labels.csv` 随机选一条记录。
2. 根据记录中的 `path` 显示验证码图片。
3. 用户输入四位数字。
4. 系统把用户输入和 `label` 字段比较。
5. 一致则验证通过，不一致则验证失败。

注意：

当前页面做的是“验证码验证”，不是“模型自动识别验证码”。

也就是说，页面判断用户输入是否正确，不需要加载模型。

模型训练和评估是项目的机器学习部分，用于研究系统能否自动识别这些验证码。

## 8. 如何理解模型训练

模型训练的目标是让神经网络自动识别验证码图片。

输入：

- 一张四位验证码图片。

输出：

- 第 1 位数字。
- 第 2 位数字。
- 第 3 位数字。
- 第 4 位数字。

训练时，每张图片都有真实标签，例如：

```text
5831
```

训练脚本会把它拆成：

```text
digit_1 = 5
digit_2 = 8
digit_3 = 3
digit_4 = 1
```

模型学习后，就可以对新验证码图片预测四个数字。

## 9. 如何理解评估结果

评估时要关注两个层次：

### 9.1 单字符准确率

单个数字是否识别正确。

例如真实标签是 `5831`，模型预测 `5837`：

- 前三位正确。
- 第四位错误。
- 单字符准确率是 3/4。

### 9.2 整体验证码准确率

四位数字必须全部正确才算对。

例如真实标签是 `5831`，模型预测 `5837`：

- 虽然前三位对了，但整张验证码仍然算错。

验证码识别更应该关注整体验证码准确率。

## 10. 如何理解 Agent 模块

Agent 是项目中的辅助分析层，不是核心识别模型。

### 10.1 数据质量审核 Agent

它帮助判断图片是否适合训练。

例如：

- 图片是不是太模糊。
- 数字是不是被遮挡。
- 标签是否可能错误。

### 10.2 错误解释 Agent

它帮助分析模型为什么错。

例如：

- `3` 被识别成 `8`。
- 干扰线遮住了关键笔画。
- 模糊太强导致数字边界不清楚。

### 10.3 训练报告 Agent

它帮助把训练结果整理成容易阅读的报告。

报告会包括：

- 数据规模。
- 模型结构。
- 训练结果。
- 错误分析。
- 下一步优化建议。

## 11. 常见问题

### 11.1 为什么网页不加载模型？

因为验证码验证页面只需要判断用户输入是否等于真实标签，不需要模型。

模型用于训练和评估自动识别能力，不用于普通用户验证页面。

### 11.2 为什么不能用 AI 生成数字图片？

项目目标要求训练必须基于真实图片。AI 生成图片可能让训练结果失真，也不符合真实数据训练原则。

### 11.3 为什么标签要保存成字符串？

因为验证码可能以 0 开头，例如 `0123`。

如果保存成数字，`0123` 会变成 `123`，标签就错了。

### 11.4 为什么有轻量页面和 Gradio 页面？

轻量页面稳定、启动快，不依赖复杂前端资源。

Gradio 页面适合快速展示机器学习 Demo。

当前默认使用轻量页面。

## 12. 建议的学习路线

如果你是第一次学习这个项目，可以按下面路线：

1. 运行 `python app.py`，先体验验证码页面。
2. 打开 `data/captcha_labels.csv`，理解图片路径和真实标签。
3. 查看 `src/data/build_captcha_dataset.py`，理解验证码如何构建。
4. 查看 `src/data/augment.py`，理解模糊和扭曲如何实现。
5. 查看 `src/models/captcha_cnn.py`，理解四输出 CNN。
6. 查看 `src/training/train_captcha.py`，理解训练流程。
7. 查看 `src/training/evaluate.py`，理解评估指标。
8. 查看 `src/llm/`，理解 Agent 辅助模块。
9. 阅读 `reports/` 中生成的报告和错误样本。

按这个顺序学习，会比直接从模型代码开始更容易理解。
