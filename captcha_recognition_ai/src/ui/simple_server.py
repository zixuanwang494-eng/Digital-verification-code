import csv
import html
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config import ALL_CAPTCHA_CSV, PROJECT_ROOT


def _load_rows() -> list[dict[str, str]]:
    if not ALL_CAPTCHA_CSV.exists():
        return []
    with ALL_CAPTCHA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _choose_captcha(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return random.choice(rows)


def _page(row: dict[str, str] | None, message: str = "") -> bytes:
    if row is None:
        body = """
        <main class="panel">
          <h1>四位数字验证码验证</h1>
          <p class="error">未找到验证码数据。请先运行数据准备命令。</p>
          <pre>python -m src.data.prepare_digits --source sklearn
python -m src.data.build_captcha_dataset --count 5000</pre>
        </main>
        """
    else:
        captcha_id = html.escape(row["captcha_id"])
        body = f"""
        <main class="panel">
          <h1>四位数字验证码验证</h1>
          <p><a href="/agent">查看 Agent 辅助模块说明</a></p>
          <img class="captcha" src="/captcha?id={captcha_id}" alt="captcha" />
          <form method="post" action="/verify">
            <input type="hidden" name="captcha_id" value="{captcha_id}" />
            <input name="answer" inputmode="numeric" maxlength="4" placeholder="请输入四位数字" autofocus />
            <button type="submit">提交验证</button>
            <a class="button secondary" href="/">刷新验证码</a>
          </form>
          <p class="message">{html.escape(message)}</p>
        </main>
        """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>四位数字验证码验证</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #1f2937;
    }}
    .panel {{
      width: min(440px, calc(100vw - 32px));
      padding: 28px;
      background: #fff;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    h1 {{
      margin: 0 0 20px;
      font-size: 22px;
    }}
    .captcha {{
      width: 320px;
      max-width: 100%;
      height: 128px;
      object-fit: contain;
      display: block;
      margin: 0 0 18px;
      border: 1px solid #cbd5e1;
      background: #fff;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: center;
    }}
    input {{
      min-width: 0;
      height: 40px;
      padding: 0 10px;
      border: 1px solid #bcc7d4;
      border-radius: 6px;
      font-size: 16px;
    }}
    button, .button {{
      height: 40px;
      padding: 0 14px;
      border: 1px solid #2563eb;
      border-radius: 6px;
      background: #2563eb;
      color: #fff;
      font-size: 14px;
      text-decoration: none;
      display: inline-grid;
      place-items: center;
      cursor: pointer;
    }}
    .secondary {{
      border-color: #9aa7b6;
      background: #fff;
      color: #1f2937;
    }}
    .message {{
      min-height: 24px;
      margin: 18px 0 0;
      font-weight: 600;
    }}
    .error {{
      color: #b91c1c;
      font-weight: 600;
    }}
    pre {{
      white-space: pre-wrap;
      background: #f8fafc;
      padding: 12px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
    }}
  </style>
</head>
<body>{body}</body>
</html>""".encode("utf-8")


def _agent_page() -> bytes:
    body = """
    <main class="panel wide">
      <h1>Agent 辅助模块说明</h1>
      <p>当前前端负责验证码展示和用户输入校验。Agent 不直接替用户作答，而是在数据、评估和报告阶段辅助项目建设。</p>

      <section>
        <h2>1. 多模态数据质量审核 Agent</h2>
        <p>脚本：<code>python -m src.llm.vision_quality_review --sample-size 20</code></p>
        <p>作用：抽样查看验证码图片，判断是否过度模糊、过度扭曲、遮挡严重或疑似标签错误。</p>
        <p>输出：<code>data/quality_reviews.csv</code> 和 <code>reports/llm_quality_review.json</code></p>
      </section>

      <section>
        <h2>2. 错误样本解释 Agent</h2>
        <p>脚本：<code>python -m src.llm.error_explainer --limit 20</code></p>
        <p>作用：读取模型评估产生的错误样本，结合图片、真实标签和模型预测解释可能的错误原因。</p>
        <p>输出：<code>reports/error_analysis.md</code> 和 <code>reports/llm_error_explanations.json</code></p>
      </section>

      <section>
        <h2>3. 训练报告生成 Agent</h2>
        <p>脚本：<code>python -m src.llm.report_generator</code></p>
        <p>作用：汇总训练历史、评估指标、错误分析和质量审核结果，生成中文 Markdown 训练报告。</p>
        <p>输出：<code>reports/training_report.md</code></p>
      </section>

      <section>
        <h2>运行边界</h2>
        <p>Agent 只做审核、解释和报告生成，不生成训练图片，不替代真实标签，也不会在验证码验证页面中自动泄露答案。</p>
      </section>

      <p><a class="button secondary" href="/">返回验证码页面</a></p>
    </main>
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent 辅助模块说明</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #1f2937;
      padding: 24px 0;
    }}
    .panel {{
      width: min(780px, calc(100vw - 32px));
      padding: 28px;
      background: #fff;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    h1 {{ margin-top: 0; font-size: 24px; }}
    h2 {{ margin-top: 24px; font-size: 18px; }}
    code {{
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .button {{
      height: 40px;
      padding: 0 14px;
      border: 1px solid #9aa7b6;
      border-radius: 6px;
      background: #fff;
      color: #1f2937;
      text-decoration: none;
      display: inline-grid;
      place-items: center;
    }}
  </style>
</head>
<body>{body}</body>
</html>""".encode("utf-8")


class CaptchaHandler(BaseHTTPRequestHandler):
    rows: list[dict[str, str]] = []

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, _page(_choose_captcha(self.rows)), "text/html; charset=utf-8")
            return
        if parsed.path == "/agent":
            self._send(200, _agent_page(), "text/html; charset=utf-8")
            return
        if parsed.path == "/captcha":
            captcha_id = parse_qs(parsed.query).get("id", [""])[0]
            row = next((item for item in self.rows if item.get("captcha_id") == captcha_id), None)
            if row is None:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            image_path = PROJECT_ROOT / row["path"]
            if not image_path.exists():
                self._send(404, b"image not found", "text/plain; charset=utf-8")
                return
            self._send(200, image_path.read_bytes(), "image/png")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/verify":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8"))
        captcha_id = data.get("captcha_id", [""])[0]
        answer = "".join(ch for ch in data.get("answer", [""])[0] if ch.isdigit())
        row = next((item for item in self.rows if item.get("captcha_id") == captcha_id), None)
        expected = str(row.get("label", "")).zfill(4) if row else ""

        if len(answer) != 4:
            message = "请输入 4 位数字。"
        elif answer == expected:
            message = "验证通过。"
        else:
            message = "验证失败，请重新输入或刷新验证码。"
        self._send(200, _page(row, message), "text/html; charset=utf-8")

    def log_message(self, format: str, *args) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 7860) -> None:
    CaptchaHandler.rows = _load_rows()
    server = ThreadingHTTPServer((host, port), CaptchaHandler)
    print(f"Running lightweight captcha app on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
