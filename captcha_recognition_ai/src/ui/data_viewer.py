import gradio as gr
import pandas as pd
from PIL import Image

from src.config import ALL_CAPTCHA_CSV, PROJECT_ROOT


def _metadata() -> pd.DataFrame:
    if not ALL_CAPTCHA_CSV.exists():
        return pd.DataFrame(columns=["captcha_id", "label", "path", "quality_status"])
    return pd.read_csv(ALL_CAPTCHA_CSV, dtype={"label": str})


def list_samples(limit: int = 50):
    df = _metadata().head(limit).copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.zfill(4)
    return df


def show_sample(captcha_id: str):
    df = _metadata()
    if captcha_id not in set(df["captcha_id"]):
        return None, "未找到样本。"
    row = df[df["captcha_id"] == captcha_id].iloc[0]
    image = Image.open(PROJECT_ROOT / row["path"]).convert("L")
    info = f"标签：{str(row['label']).zfill(4)}\n质量状态：{row.get('quality_status', 'unknown')}"
    return image, info


with gr.Blocks(title="验证码数据集浏览") as data_viewer:
    gr.Markdown("# 验证码数据集浏览")
    table = gr.Dataframe(value=list_samples, interactive=False)
    sample_id = gr.Textbox(label="captcha_id")
    image = gr.Image(type="pil", label="样本图片")
    info = gr.Textbox(label="样本信息")
    gr.Button("查看样本").click(show_sample, inputs=sample_id, outputs=[image, info])
