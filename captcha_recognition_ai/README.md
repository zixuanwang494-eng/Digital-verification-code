# 四位数字验证码识别与智能评估系统

本项目是在原始单数字 CNN 识别项目基础上重新组织的新项目，用于实现四位数字验证码识别、真实数字数据集管理、模型训练评估，以及可选的大模型辅助质检、错误解释和训练报告生成。

## 核心原则

- 训练基础数据必须来自真实数字图片数据集。
- 不使用程序随机绘制数字作为主要训练数据。
- 不使用 AI 生成图片冒充训练数据。
- 图像增强只作用于真实数字图片。
- 大模型只用于辅助审核、解释和报告生成，不替代真实标签。

## 推荐运行流程

```bash
cd captcha_recognition_ai
pip install -r requirements.txt

python -m src.data.prepare_digits --source sklearn
python -m src.data.build_captcha_dataset --count 5000
python -m src.data.dataset_manager stats
python -m src.training.train_captcha --epochs 10
python -m src.training.evaluate --model models/captcha_cnn.keras
python app.py
```

`app.py` 默认启动轻量网页，不加载 TensorFlow 模型，也不依赖 Gradio 前端资源。访问终端显示的本地地址即可，通常是 `http://127.0.0.1:7860`。

页面中的“查看 Agent 辅助模块说明”链接会说明三个 Agent 在项目中的作用：数据质量审核、错误样本解释、训练报告生成。

如需使用 Gradio 版本，可以运行：

```bash
python -c "from src.ui.login_demo import demo; demo.launch()"
```

如需单独启动轻量版本，也可以运行：

```bash
python -c "from src.ui.simple_server import run; run()"
```

## 可选大模型能力

如需使用大模型辅助能力，请设置环境变量：

```bash
OPENAI_API_KEY=你的 API Key
OPENAI_VISION_MODEL=你的视觉模型名称
OPENAI_TEXT_MODEL=你的文本模型名称
```

然后可以运行：

```bash
python -m src.llm.vision_quality_review --sample-size 20
python -m src.llm.error_explainer --limit 20
python -m src.llm.report_generator
```

如果未配置 API Key，这些模块会生成本地占位报告，不影响普通训练、评估和界面使用。

## 文档

详细项目说明见 [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md)。
