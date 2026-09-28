"""deepseek-eyes MCP Server — 给 DeepSeek 装上眼睛

通过通义千问VL (Qwen-VL via ModelScope) 为无视觉能力的文本模型
提供图片理解能力。支持剪贴板直接读取和文件路径两种方式。

兼容 MCP SDK v1 (Server + list_tools/call_tool) 和 v2 (MCPServer + @tool)。
启动时自动检测可用版本，无需手动配置。
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import os
from pathlib import Path
from typing import Any

import aiofiles
from openai import AsyncOpenAI

from .clipboard import ClipboardError, save_clipboard_image

SERVER_NAME = "deepseek-eyes"
SERVER_VERSION = "1.0.0"

# ── MCP 版本检测 ──────────────────────────────────────────────
# 优先 v2 (MCPServer)，回退 v1 (Server + stdio_server)，两者都失败则报错

_MCP_V2 = False

try:
    from mcp.server import MCPServer

    _MCP_V2 = True
except ImportError:
    pass

if not _MCP_V2:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError(
            "无法导入 MCP SDK (v1 或 v2)。请安装: pip install 'mcp>=1.0.0'"
        )

# ── ModelScope API 配置 ───────────────────────────────────────

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VISION_MODEL = os.environ.get("VISION_MODEL", DEFAULT_MODEL)

# 安全检查
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"RIFF",
    b"BM",
)

API_KEY_HELP = (
    "❌ 未设置 MODELSCOPE_API_KEY 环境变量。\n\n"
    "获取免费 API Key（每天2000次，单模型500次）：\n"
    "1. 打开 https://modelscope.cn/my/myaccesstoken\n"
    "2. 登录 → 首次使用需绑定阿里云账号\n"
    "3. 点击「新建访问令牌」→ 命名 → 生成 → 复制\n"
    "4. ⚠️ 令牌格式为 ms-xxxxxxxx，使用时去掉 ms- 前缀！\n"
    "5. 将去掉前缀后的 Key 设置到 MCP 配置的 env 中\n\n"
    "MODELSCOPE_API_KEY not set. "
    "Get a free key at https://modelscope.cn/my/myaccesstoken "
    "(2000 calls/day, remove ms- prefix)."
)

PROMPTS: dict[str, str] = {
    "analyze": (
        "请详细描述这张图片的内容。包括所有相关元素、上下文，"
        "以及任何对看不到图片的人有用的信息。"
    ),
    "extract_text": (
        "提取这张图片中的全部文字。只返回文字内容，保留排版和换行，"
        "不做任何评论。"
    ),
    "describe_ui": (
        "分析这张 UI 截图。描述：1) 整体布局 2) 组件（按钮、表单、导航、输入框）"
        "3) 可见文字和标签 4) 状态（错误提示、激活标签页、弹窗等）。"
    ),
    "diagnose_error": (
        "分析这张错误截图。返回：1) 精确的错误信息 2) 可能的原因 "
        "3) 具体的修复步骤 4) 如何避免再次发生。"
    ),
    "understand_diagram": (
        "解读这张图表。返回：1) 图表类型 2) 组成部分及其作用 "
        "3) 关系/流程 4) 整体目的。"
    ),
    "analyze_chart": (
        "分析这张数据图表。返回：1) 图表类型 2) 坐标轴和标签 "
        "3) 关键趋势 4) 值得注意的数据点 5) 洞察。"
    ),
    "code_from_screenshot": (
        "从这张截图中提取全部代码。返回：1) 编程语言 "
        "2) 格式化的代码块，保留缩进。"
    ),
}

# ── Vision Client（版本无关的共享逻辑）──────────────────────────


def _validate_image_path(path_str: str) -> Path:
    """校验图片路径，拒绝非图片文件和超大文件。"""
    p = Path(path_str).resolve()
    if not p.is_file():
        raise ValueError(f"不是一个文件: {path_str}")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"拒绝读取 '{p.suffix}' —— 仅允许图片格式 "
            f"({', '.join(sorted(ALLOWED_EXTENSIONS))})。"
        )
    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大: {size} 字节 (最大 {MAX_IMAGE_BYTES})。")
    return p


def _validate_magic(data: bytes) -> None:
    """校验文件魔数。"""
    if not any(data.startswith(m) for m in IMAGE_MAGIC_PREFIXES):
        raise ValueError("文件内容不像是支持的图片格式。")


class VisionClient:
    """通义千问VL 视觉客户端 (via ModelScope OpenAI-compatible API)"""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=MODELSCOPE_BASE_URL)

    async def analyze(self, image_path: str, prompt: str) -> str:
        p = _validate_image_path(image_path)
        async with aiofiles.open(p, "rb") as f:
            data = await f.read()
        _validate_magic(data)
        b64 = base64.b64encode(data).decode("utf-8")

        response = await self.client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


_vision_client: VisionClient | None = None


def _get_client() -> VisionClient | None:
    """获取 VisionClient 实例。未配置 API Key 时返回 None。"""
    global _vision_client
    if _vision_client is None:
        api_key = os.environ.get("MODELSCOPE_API_KEY")
        if api_key:
            _vision_client = VisionClient(api_key)
    return _vision_client


async def _run(prompt_key: str, image_path: str, override: str | None = None) -> str:
    """执行图片分析（文件路径模式）。"""
    client = _get_client()
    if client is None:
        return API_KEY_HELP
    try:
        prompt = override or PROMPTS[prompt_key]
        return await client.analyze(image_path, prompt)
    except Exception as e:
        return f"错误: {e}"


async def _run_clipboard(prompt_key: str, override: str | None = None) -> str:
    """执行图片分析（剪贴板模式）。"""
    client = _get_client()
    if client is None:
        return API_KEY_HELP
    try:
        path = save_clipboard_image()
    except ClipboardError as e:
        return f"剪贴板错误: {e}"
    try:
        prompt = override or PROMPTS[prompt_key]
        return await client.analyze(path, prompt)
    except Exception as e:
        return f"错误: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── 工具元数据（版本无关的定义）────────────────────────────────
# 每个工具: (name, description, handler, has_prompt_param)
# handler 是共享的业务逻辑函数，不管 MCP 版本如何都调同一个


async def _analyze_clipboard(prompt: str | None = None) -> str:
    return await _run_clipboard("analyze", prompt)


async def _extract_text_from_clipboard() -> str:
    return await _run_clipboard("extract_text")


async def _describe_ui_from_clipboard() -> str:
    return await _run_clipboard("describe_ui")


async def _diagnose_error_from_clipboard() -> str:
    return await _run_clipboard("diagnose_error")


async def _code_from_clipboard() -> str:
    return await _run_clipboard("code_from_screenshot")


async def _analyze_image(image_path: str, prompt: str | None = None) -> str:
    return await _run("analyze", image_path, prompt)


async def _extract_text(image_path: str) -> str:
    return await _run("extract_text", image_path)


async def _describe_ui(image_path: str) -> str:
    return await _run("describe_ui", image_path)


async def _diagnose_error(image_path: str) -> str:
    return await _run("diagnose_error", image_path)


async def _understand_diagram(image_path: str) -> str:
    return await _run("understand_diagram", image_path)


async def _analyze_chart(image_path: str) -> str:
    return await _run("analyze_chart", image_path)


async def _code_from_screenshot(image_path: str) -> str:
    return await _run("code_from_screenshot", image_path)


# (name, description, handler, extra_params_schema)
_TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "analyze_clipboard",
        "description": (
            "读取系统剪贴板中的图片并分析。"
            "当用户说'看看这个'、'剪贴板里有什么'、或粘贴截图时使用。"
            "可选参数 prompt 可自定义提问。"
            " / Analyze the image in system clipboard."
        ),
        "handler": _analyze_clipboard,
        "params": {"prompt": {"type": "string", "description": "自定义问题 / Custom question."}},
        "required": [],
    },
    {
        "name": "extract_text_from_clipboard",
        "description": "从剪贴板图片中提取文字(OCR) / Extract text from clipboard image (OCR).",
        "handler": _extract_text_from_clipboard,
        "params": {},
        "required": [],
    },
    {
        "name": "describe_ui_from_clipboard",
        "description": (
            "描述剪贴板中 UI 截图的布局、组件和状态。"
            " / Describe UI from clipboard screenshot."
        ),
        "handler": _describe_ui_from_clipboard,
        "params": {},
        "required": [],
    },
    {
        "name": "diagnose_error_from_clipboard",
        "description": (
            "诊断剪贴板中错误截图的原因和修复方案。"
            " / Diagnose error screenshot from clipboard."
        ),
        "handler": _diagnose_error_from_clipboard,
        "params": {},
        "required": [],
    },
    {
        "name": "code_from_clipboard",
        "description": (
            "从剪贴板代码截图中提取可编辑的代码。"
            " / Extract code from clipboard screenshot."
        ),
        "handler": _code_from_clipboard,
        "params": {},
        "required": [],
    },
    {
        "name": "analyze_image",
        "description": (
            "分析磁盘上的图片文件。支持 png/jpg/gif/webp/bmp。"
            " / Analyze an image file on disk."
        ),
        "handler": _analyze_image,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
            "prompt": {"type": "string", "description": "自定义问题 / Custom question."},
        },
        "required": ["image_path"],
    },
    {
        "name": "extract_text",
        "description": "从磁盘图片中提取文字(OCR) / OCR an image file on disk.",
        "handler": _extract_text,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
    {
        "name": "describe_ui",
        "description": "描述磁盘上 UI 截图文件的布局、组件和状态 / Describe a UI screenshot file on disk.",
        "handler": _describe_ui,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
    {
        "name": "diagnose_error",
        "description": "诊断磁盘上错误截图文件的原因和修复方案 / Diagnose an error screenshot file on disk.",
        "handler": _diagnose_error,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
    {
        "name": "understand_diagram",
        "description": "解读流程图/架构图等图表文件 / Interpret a diagram image file on disk.",
        "handler": _understand_diagram,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
    {
        "name": "analyze_chart",
        "description": "分析数据图表文件中的趋势和洞察 / Analyze a chart image file on disk.",
        "handler": _analyze_chart,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
    {
        "name": "code_from_screenshot",
        "description": "从磁盘代码截图文件中提取代码 / Extract code from a screenshot file on disk.",
        "handler": _code_from_screenshot,
        "params": {
            "image_path": {"type": "string", "description": "图片文件的绝对路径 / Absolute path to the image file."},
        },
        "required": ["image_path"],
    },
]


def _make_input_schema(params: dict, required: list[str]) -> dict:
    """构建 JSON Schema。"""
    if not params:
        return {"type": "object", "properties": {}, "required": []}
    return {
        "type": "object",
        "properties": params,
        "required": required,
    }


# ── MCP 注册层 ────────────────────────────────────────────────


def _register_v2() -> MCPServer:
    """MCP SDK v2: 用 MCPServer.add_tool() 注册。schema 由 type hints 自动生成。"""
    mcp = MCPServer(SERVER_NAME)
    for tool in _TOOL_REGISTRY:
        mcp.add_tool(
            tool["handler"],
            name=tool["name"],
            description=tool["description"],
        )
    return mcp


def _register_v1():
    """MCP SDK v1: 用 Server + list_tools/call_tool 装饰器注册。"""
    server = Server(SERVER_NAME)

    # 构建 v1 Tool 列表
    v1_tools = [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=_make_input_schema(t["params"], t["required"]),
        )
        for t in _TOOL_REGISTRY
    ]

    # 按 name 索引 handler
    handler_map = {t["name"]: t["handler"] for t in _TOOL_REGISTRY}

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return v1_tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = handler_map.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"未知工具: {name}")]
        try:
            sig = inspect.signature(handler)
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in arguments:
                    kwargs[param_name] = arguments[param_name]
            text = await handler(**kwargs)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            return [TextContent(type="text", text=f"错误: {e}")]

    return server


# ── 启动入口 ──────────────────────────────────────────────────

if _MCP_V2:
    _mcp = _register_v2()

    def run() -> None:
        """入口: python -m deepseek_eyes"""
        _mcp.run()

else:
    _server = _register_v1()

    def run() -> None:
        """入口: python -m deepseek_eyes"""
        asyncio.run(_run_v1())

    async def _run_v1() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await _server.run(
                read_stream, write_stream, _server.create_initialization_options()
            )


if __name__ == "__main__":
    run()
