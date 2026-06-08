import random
import gradio as gr
import pandas as pd
from PIL import Image

from src.config import ALL_CAPTCHA_CSV, PROJECT_ROOT


def _load_metadata() -> pd.DataFrame:
    if not ALL_CAPTCHA_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ALL_CAPTCHA_CSV, dtype={"label": str})


def _random_captcha() -> tuple[Image.Image | None, str, str]:
    metadata = _load_metadata()
    if metadata.empty:
        return None, "", "请先运行数据准备脚本生成验证码数据集。"
    row = metadata.sample(1, random_state=random.randint(0, 10_000_000)).iloc[0]
    image = Image.open(PROJECT_ROOT / row["path"]).convert("L")
    label = str(row["label"]).zfill(4)
    return image, label, "请输入图片中的四位数字。"


def refresh_captcha():
    image, label, message = _random_captcha()
    return image, label, "", message


def verify(user_input: str, label: str):
    normalized = "".join(ch for ch in (user_input or "") if ch.isdigit())
    expected = str(label).zfill(4)
    if len(normalized) != 4:
        return "请输入 4 位数字。"
    if normalized == expected:
        return "验证通过。"
    return "验证失败，请重新输入或刷新验证码。"


initial_image, initial_label, initial_message = _random_captcha()


with gr.Blocks(title="四位数字验证码验证") as demo:
    gr.Markdown("# 四位数字验证码验证")
    gr.Markdown("验证码图片来自真实数字图片构建的数据集，并经过轻度模糊、扭曲和干扰处理。")
    gr.Markdown(
        "Agent 辅助模块用于后台数据质量审核、错误样本解释和训练报告生成，"
        "不会在验证码验证页面中自动作答或泄露标签。"
    )

    hidden_label = gr.State(initial_label)
    captcha_image = gr.Image(value=initial_image, label="验证码", type="pil", height=140)
    user_input = gr.Textbox(label="输入验证码", placeholder="请输入四位数字", max_lines=1)
    message = gr.Textbox(value=initial_message, label="结果", interactive=False)

    with gr.Row():
        submit = gr.Button("提交验证", variant="primary")
        refresh = gr.Button("刷新验证码")

    refresh.click(fn=refresh_captcha, inputs=None, outputs=[captcha_image, hidden_label, user_input, message])
    submit.click(fn=verify, inputs=[user_input, hidden_label], outputs=message)
