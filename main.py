"""AstrBot plugin for GPT image generation models.

The plugin exposes a single API config block. Internally it keeps two
parameter profiles (``standard`` and ``flexible``) that map to OpenAI's
strict Images API and the looser webpage-reverse / 2api / ToAPIs dialect.
A profile is picked per request from cache or heuristics; on a parameter
or vague upstream compatibility error the request is retried once with the
other profile, and the successful profile is cached per
``(base_url, model, operation)``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.core.message.components import Node, Nodes
from astrbot.api.platform import MessageType
from astrbot.api.provider import Provider, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.utils.quoted_message import extract_quoted_message_images


PLUGIN_ID = "astrbot_plugin_gpt_image_2"
TOOL_NAME = "generate_image_with_gpt_image_2"
PLUGIN_DISPLAY_NAME = "GPT Image · 龙绘"

MAX_GPT_IMAGE_REFERENCE_IMAGES = 16
MAX_REFERENCE_IMAGE_BYTES = 50 * 1024 * 1024

# GPT image providers do not all accept the same size syntax. Keep the
# selectable pixel presets limited to sizes verified against the current
# upstream, while retaining the official GPT Image defaults.
VALID_OFFICIAL_IMAGE_SIZES = {"1024x1024", "1536x1024", "1024x1536"}
VALID_UPSTREAM_IMAGE_SIZES = VALID_OFFICIAL_IMAGE_SIZES | {
    "2560x1440",
    "1440x2560",
    "2880x2880",
    "3840x2160",
    "2160x3840",
}
REFERENCE_SQUARE_RATIO_MIN = 1 / 1.2
REFERENCE_SQUARE_RATIO_MAX = 1.2
REFERENCE_LANDSCAPE_SIZE = "1536x1024"
REFERENCE_PORTRAIT_SIZE = "1024x1536"
REFERENCE_SQUARE_SIZE = "1024x1024"
VALID_ASPECT_RATIOS = {
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "21:9",
    "9:21",
}
VALID_RESOLUTIONS = {"auto", "1k", "2k"}
VALID_QUALITIES = {"auto", "low", "medium", "high"}
VALID_OUTPUT_FORMATS = {"png", "jpeg", "webp", "gif"}
VALID_BACKGROUNDS = {"auto", "transparent", "opaque"}
VALID_INPUT_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
TRANSIENT_HTTP_STATUSES = {408, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526}
NETWORK_RETRY_BASE_DELAY_SECONDS = 1.5
TRACE_PROMPT_PREVIEW_LIMIT = 900
TRACE_QUERY_TEXT_LIMIT = 1300
HTTP_TRACE_HEADER_NAMES = {
    "x-request-id",
    "x-trace-id",
    "x-correlation-id",
    "x-ratelimit-request-id",
    "openai-processing-ms",
    "cf-ray",
}

DIRECT_REFERENCE_PROMPT_ONLY_PATTERNS = [
    r"(翻译|译成|翻成|翻译成|translate|translation).*(中文|英文|日文|日语|韩文|韩语|法文|德文|俄文|to|into)",
    r"(这段话|这段|这句话|这句|这段文字|这段英文|这段日文|这段prompt|thistext|thissentence).*(翻译|译成|翻成|translate)",
    r"^(翻译|译成|翻成|translate).+",
    r"(提示词|prompt).*(写|改|优化|润色|翻译|扩写|整理)",
    r"(生成|写|改|优化|润色|翻译|扩写|整理).*(提示词|prompt)",
    r"(怎么|如何|为什么|什么|啥|能不能|可以吗|参数|设置|教程|建议|方案|分析|评价).*(画|生成|生图|出图|改图|图片|图像)",
    r"(how|what|why|help|guide|tutorial|advice|suggest|analy[sz]e|review).*(draw|generate|create|edit|image|picture|photo|prompt)",
    r"(prompt).*(write|improve|rewrite|translate|optimi[sz]e|polish)",
    r"(write|improve|rewrite|translate|optimi[sz]e|polish).*(prompt)",
]

DIRECT_REFERENCE_EDIT_PATTERNS = [
    r"(改图|修图|p图|重绘|重画|重新画|扩图|扩展|延展|外扩|补图|补全|补完|补齐|抠图)",
    r"(重画|重绘|重新画|改画|转绘|画成|绘成|渲染成|改成|改为|变成|变为|转成|转为|做成).*(风格|画风|材质|质感|效果|样子|手办|动漫|写实|3d|c4d|像素|水彩|油画|产品渲染|高清渲染)",
    r"(改|修|换|替换|去掉|删除|加上|添加|合成|参考|照着|补全|补完|补齐|扩展|延展|外扩).*(这张|上图|图片|图里|图中|照片|背景|主体|人物|文字|全身照)",
    r"(这张|上图|图片|图里|图中|照片|背景|主体|人物|文字|全身照).*(改|修|换|替换|去掉|删除|加上|添加|合成|重画|重绘|重新画|改画|画成|绘成|渲染成|补全|补完|补齐|扩展|延展|外扩|改成|改为|变成|变为|转成|转为|做成|高清渲染|c4d)",
    r"(给|帮|帮我|请).*(这张|上图|图片|图里|图中|照片|背景|主体|人物|文字|全身照).*(改|修|换|替换|去掉|删除|加上|添加|合成|重画|重绘|重新画|改画|画成|绘成|渲染成|补全|补完|补齐|扩展|延展|外扩|改成|改为|变成|变为|转成|转为|做成|高清渲染|c4d)",
    r"(把|将).*(图|图片|照片|背景|主体|人物|文字).*(改|修|换|替换|去掉|删除|加上|添加|合成|重画|重绘|重新画|改画|画成|绘成|渲染成|补全|补完|补齐|扩展|延展|外扩|变成|变为|转成|做成|高清渲染|c4d)",
    r"(edit|modify|change|replace|remove|delete|add|combine|merge|redraw|expand|outpaint|cutout).*(this|the)?(image|picture|photo|background|subject|person|text)",
    r"(this|the)?(image|picture|photo|background|subject|person|text).*(edit|modify|change|replace|remove|delete|add|combine|merge|redraw|expand|outpaint|cutout)",
    r"(画像|写真|背景|人物|文字).*(編集|修正|変更|置換|削除|追加|合成)",
]

DIRECT_REFERENCE_GENERATION_PATTERNS = [
    r"(参考|照着|根据|按照|按|用).*(这张|上图|图片|图|照片).*(画|绘制|生成|生图|出图|做|设计)",
    r"(画|绘制|生成|生图|出图|做|设计).*(参考|照着|根据|按照|按|用).*(这张|上图|图片|图|照片)",
    r"(reference|use|basedon|from).*(this|the)?(image|picture|photo).*(draw|generate|create|make|paint|design)",
    r"(draw|generate|create|make|paint|design).*(reference|use|basedon|from).*(this|the)?(image|picture|photo)",
]

DIRECT_TEXT_GENERATION_PATTERNS = [
    r"(画|绘制|生成|生图|出图|出一张|做一张|做张|做一个|做个|来一张|来张).*(图|图片|图像|画|海报|头像|壁纸|图标|logo|贴纸|表情|插画|漫画|照片|场景|全身照)",
    r"(再生成|重新生成|重生成|继续生成|生成).*(一遍|一次|安全|动漫|写实|风格|画风|版|图|图片|插画|场景)",
    r"(再画|重新画).+",
    r"(再来|重新来).*(一遍|一次|一张|张|图|图片|版|插画|场景)",
    r"(画|绘制)(一张|一个|个|只|幅|张).+",
    r"(生成|做|设计|制作)(一张|一个|个|张).+",
    r"(帮我|给我|替我).*(画|绘制|生成|生图|出图|做一张|做张|做一个|做个|来一张|来张)",
    r"(设计|制作|做).*(一张|一个|个).*(图|图片|图像|海报|头像|壁纸|图标|logo|贴纸|表情|插画|漫画)",
    r"^(画|绘制|生成|生图|出图|做一张|做张|做一个|做个|来一张|来张)",
    r"(draw|generate|create|make|paint|design).*(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
    r"(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration).*(draw|generate|create|make|paint|design)",
    r"(can|could|please|pls|wouldyou|canyou|couldyou|helpme).*(draw|paint).+",
    r"(can|could|please|pls|wouldyou|canyou|couldyou|helpme).*(generate|create|make|design).*(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
    r"^(draw|paint)(a|an|the)?.+",
    r"^(draw|generate|create|make|paint|design)(an?|the)?(image|picture|photo|poster|avatar|wallpaper|icon|logo|sticker|illustration|comic|scene)",
    r"(画像|イラスト|写真|ポスター|アイコン|壁紙|ロゴ).*(描いて|生成|作成|作って|描く)",
    r"(描いて|生成|作成|作って).*(画像|イラスト|写真|ポスター|アイコン|壁紙|ロゴ)",
    r"(描いて|描く|生成して|作って|作成して)",
]

STRICT_DIRECT_TEXT_GENERATION_PATTERNS = [
    r"^(画|绘制)(一张|一个|个|只|幅|张).+",
    r"^(生成|生图|出图|出一张|做一张|做张|做一个|做个|来一张|来张).+",
    r"^(帮我|给我|替我|请|麻烦|帮忙).*(画|绘制|生成|生图|出图|出一张|做一张|做张|做一个|做个|来一张|来张).+",
    r"^(我要|我想要|想要|要).*(一张|张|个).*(图|图片|图像|海报|头像|壁纸|图标|logo|贴纸|表情|插画|照片|场景|全身照)",
    r"(按|按照|根据|基于).*(描述|文字|文案|内容|提示词|prompt).*(画|绘制|生成|生一张|生张|生个|生图|出图|做|设计).*(图|图片|图像|画|插画|场景)",
]

DIRECT_TEXT_DISCUSSION_GUARD_PATTERNS = [
    r"(动画|漫画|小说|剧情|章节|集数|话数|篇章|更新).*(进度|同样|一样|一个进度|到哪|到哪儿|到哪里|追到|更到|第几|多少)",
    r"(进度|同样|一样|一个进度|到哪|到哪儿|到哪里|第几|多少).*(动画|漫画|小说|剧情|章节|集数|话数|篇章|更新)",
    r"^(不是)?(我)?(是)?(问|问你|请问|想问).+",
]

DIRECT_TEXT_QUESTION_MARKERS = (
    "？",
    "?",
    "吗",
    "嘛",
    "么",
    "怎么",
    "如何",
    "为什么",
    "什么",
    "啥",
    "哪",
    "几",
    "多少",
    "是否",
    "是不是",
    "有没有",
    "能不能",
    "可以吗",
    "行不行",
    "区别",
    "进度",
)

DESCRIPTION_REFERENCE_GENERATION_PATTERNS = [
    r"(按|按照|根据|基于|用|依照).*(这张图|这张图片|这个图|这个图片|这张|这个|上图|引用|图里|图中|图片里|图片中).*(描述|文字|文案|内容|提示词|prompt).*(画|绘制|生成|生一张|生张|生个|生图|出图|做|设计)",
    r"(画|绘制|生成|生一张|生张|生个|生图|出图|做|设计).*(按|按照|根据|基于|用|依照).*(这张图|这张图片|这个图|这个图片|这张|这个|上图|引用|图里|图中|图片里|图片中).*(描述|文字|文案|内容|提示词|prompt)",
    r"(按|按照|根据|基于|用|依照).*(这个|这段|上面|引用|图里|图中|图片里|图片中).*(描述|文字|文案|内容|提示词|prompt).*(画|绘制|生成|生图|出图|做|设计)",
    r"(画|绘制|生成|生图|出图|做|设计).*(按|按照|根据|基于|用|依照).*(这个|这段|上面|引用|图里|图中|图片里|图片中).*(描述|文字|文案|内容|提示词|prompt)",
    r"(按|按照|根据|基于|用|依照).*(描述|文字|文案|内容|提示词|prompt).*(画|绘制|生成|生一张|生张|生个|生图|出图|做|设计).*(图|图片|图像|画|插画|场景)",
    r"(read|ocr|extract).*(text|description|prompt).*(image|picture|photo).*(draw|generate|create|make)",
    r"(draw|generate|create|make).*(from|basedon|accordingto).*(text|description|prompt).*(image|picture|photo)",
]

DIALOG_CONTEXT_TEXT_PATTERNS = [
    r"(自拍|自画像|人设|人格|本体|selfie|persona|自撮り)",
    r"(画|绘制|生成|生图|出图|做|设计|制作|来).{0,8}(你自己|妳自己|你|妳|你的|妳的|本体|本人|人设|人格|设定|形象|自拍|自画像)",
    r"(你自己|妳自己|你的|妳的|本体|本人|人设|人格|设定|形象).{0,12}(图|图片|照片|头像|壁纸|海报|自拍|自画像|全身照|半身照)",
    r"(自拍|自画像).{0,8}(你|妳|bot|机器人|本体|人设|形象)",
    r"(draw|generate|create|make|paint|design).{0,24}(yourself|yourselfie|selfie|yourpersona|yourcharacter|youravatar|yourfullbody|yourown)",
    r"(yourself|yourselfie|selfie|yourpersona|yourcharacter|youravatar|yourfullbody|yourown).{0,24}(image|picture|photo|portrait|avatar|poster|selfie)",
    r"(描いて|生成して|作って).{0,8}(君|あなた|自分|本人|人格|設定|自撮り)",
]


def _safe_event_text(event: AstrMessageEvent) -> str:
    try:
        return str(getattr(event, "message_str", "") or event.get_message_str() or "").strip()
    except Exception:
        return str(getattr(event, "message_str", "") or "").strip()


def _normalized_request_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _strict_direct_text_allows_image_generation(normalized: str) -> bool:
    return any(
        re.search(pattern, normalized)
        for pattern in STRICT_DIRECT_TEXT_GENERATION_PATTERNS
    )


def _direct_text_looks_like_discussion_or_question(normalized: str) -> bool:
    if any(
        re.search(pattern, normalized)
        for pattern in DIRECT_TEXT_DISCUSSION_GUARD_PATTERNS
    ):
        return True
    return any(marker in normalized for marker in DIRECT_TEXT_QUESTION_MARKERS)


def _direct_reference_text_allows_image_edit(text: str) -> bool:
    if _text_requests_gif(text):
        return False
    normalized = _normalized_request_text(text)
    if not normalized:
        return False
    if normalized.startswith(("/改图", "/生图")):
        return False
    if any(re.search(pattern, normalized) for pattern in DIRECT_REFERENCE_PROMPT_ONLY_PATTERNS):
        return False
    if _reference_context_text_prefers_edit(normalized):
        return True
    return any(
        re.search(pattern, normalized)
        for pattern in DIRECT_REFERENCE_EDIT_PATTERNS + DIRECT_REFERENCE_GENERATION_PATTERNS
    )


def _direct_text_allows_image_generation(text: str) -> bool:
    normalized = _normalized_request_text(text)
    if not normalized:
        return False
    if normalized.startswith(("/改图", "/生图")):
        return False
    if any(re.search(pattern, normalized) for pattern in DIRECT_REFERENCE_PROMPT_ONLY_PATTERNS):
        return False
    if _strict_direct_text_allows_image_generation(normalized):
        return True
    if _direct_text_looks_like_discussion_or_question(normalized):
        return False
    return any(re.search(pattern, normalized) for pattern in DIRECT_TEXT_GENERATION_PATTERNS)


def _reference_context_text_prefers_edit(text: str) -> bool:
    normalized = _normalized_request_text(text)
    if not normalized:
        return False
    patterns = [
        r"(\u91cd\u753b|\u91cd\u7ed8|\u91cd\u65b0\u753b|\u91cd\u65b0\u7ed8|\u6539\u56fe|\u4fee\u56fe|\u6539\u6210|\u6539\u4e3a|\u6362\u6210|\u66ff\u6362|\u53d8\u6210|\u53d8\u4e3a|\u53bb\u6389|\u5220\u9664|\u79fb\u9664|\u52a0\u4e0a|\u6dfb\u52a0|\u4fdd\u7559|\u8865\u5168|\u6269\u56fe|\u6269\u5c55|\u5916\u6269)",
        r"(\u53c2\u8003|\u7167\u7740|\u6839\u636e|\u6309\u7167|\u4f9d\u7167).*(\u8fd9\u5f20|\u8fd9\u4e2a|\u4e0a\u56fe|\u56fe|\u56fe\u7247|\u7167\u7247|\u53c2\u8003\u56fe)",
        r"(\u8fd9\u5f20|\u8fd9\u4e2a|\u4e0a\u56fe|\u56fe|\u56fe\u7247|\u7167\u7247|\u53c2\u8003\u56fe).*(\u91cd\u753b|\u91cd\u7ed8|\u6539|\u6362|\u53bb\u6389|\u5220\u9664|\u79fb\u9664|\u4fdd\u7559|\u8865\u5168|\u6269\u56fe|\u6269\u5c55)",
        r"(\u5168\u8eab\u7167|\u534a\u8eab\u7167|\u8fd1\u666f|\u8fdc\u666f|\u7279\u5199).*(\u53bb\u6389|\u5220\u9664|\u79fb\u9664|\u4fdd\u7559|\u91cd\u753b|\u91cd\u7ed8|\u6539|\u6362)",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _description_reference_text_allows_image_generation(text: str) -> bool:
    normalized = _normalized_request_text(text)
    if not normalized:
        return False
    if any(re.search(pattern, normalized) for pattern in DIRECT_REFERENCE_PROMPT_ONLY_PATTERNS):
        return False
    return any(re.search(pattern, normalized) for pattern in DESCRIPTION_REFERENCE_GENERATION_PATTERNS)


def _text_requests_gif(text: str) -> bool:
    return bool(re.search(r"(?<![a-z])gif(?![a-z])|动图|动态表情|动态贴纸|动画", str(text or ""), flags=re.I))


def _text_generation_needs_dialog_context(text: str) -> bool:
    """Leave contextual subjects and animation parameters to the dialogue."""
    if _text_requests_gif(text):
        return True
    normalized = _normalized_request_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in DIALOG_CONTEXT_TEXT_PATTERNS)


def _message_chain_has_image_or_reply(chain: Any) -> bool:
    if isinstance(chain, Image):
        return True
    if isinstance(chain, Reply):
        return True
    if not isinstance(chain, (list, tuple)):
        return False

    for component in chain:
        if isinstance(component, (Image, Reply)):
            return True
        nested_chain = getattr(component, "chain", None)
        if nested_chain and _message_chain_has_image_or_reply(nested_chain):
            return True
        nested_content = getattr(component, "content", None)
        if nested_content and _message_chain_has_image_or_reply(nested_content):
            return True
        nested_nodes = getattr(component, "nodes", None)
        if nested_nodes:
            for node in nested_nodes:
                if _message_chain_has_image_or_reply(getattr(node, "content", None)):
                    return True
    return False


def _message_chain_has_image(chain: Any) -> bool:
    if isinstance(chain, Image):
        return True
    if not isinstance(chain, (list, tuple)):
        return False

    for component in chain:
        if isinstance(component, Image):
            return True
        nested_chain = getattr(component, "chain", None)
        if nested_chain and _message_chain_has_image(nested_chain):
            return True
        nested_content = getattr(component, "content", None)
        if nested_content and _message_chain_has_image(nested_content):
            return True
        nested_nodes = getattr(component, "nodes", None)
        if nested_nodes:
            for node in nested_nodes:
                if _message_chain_has_image(getattr(node, "content", None)):
                    return True
    return False


def _message_chain_mentions_bot(chain: Any, self_id: str) -> bool:
    if not self_id:
        return False
    if isinstance(chain, At):
        return str(getattr(chain, "qq", "")).strip() == self_id
    if not isinstance(chain, (list, tuple)):
        return False

    for component in chain:
        if isinstance(component, At) and str(getattr(component, "qq", "")).strip() == self_id:
            return True
    return False


def _group_message_mentions_this_bot(event: AstrMessageEvent) -> bool:
    try:
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return True
    except Exception:
        return True

    try:
        self_id = str(event.get_self_id() or "").strip()
    except Exception:
        self_id = ""
    if not self_id:
        return False

    try:
        message_chain = event.get_messages()
    except Exception:
        message_chain = getattr(getattr(event, "message_obj", None), "message", [])
    return _message_chain_mentions_bot(message_chain, self_id)


def _config_bool(cfg: Any, section: str, key: str, default: bool) -> bool:
    try:
        section_value = cfg.get(section, {}) if hasattr(cfg, "get") else {}
        value = section_value.get(key, default) if hasattr(section_value, "get") else default
    except Exception:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "启用", "开启"}
    return bool(value)


def _plan_command_payload(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()

    if lowered.startswith("/plan"):
        if len(raw) == 5:
            return ""
        next_char = raw[5]
        if next_char.isspace() or next_char in {":", "：", "["} or not next_char.isascii():
            return raw[5:].lstrip()
        return None

    if lowered.startswith("plan"):
        if len(raw) == 4:
            return ""
        next_char = raw[4]
        if next_char.isspace() or next_char in {":", "：", "["} or not next_char.isascii():
            return raw[4:].lstrip()
    return None


def _strip_image_placeholders(text: str) -> str:
    return re.sub(r"(?:\[图片\]|\[image\])\s*", "", str(text or ""), flags=re.IGNORECASE).strip()


class DirectReferenceImageEditFilter(filter.CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        if not _config_bool(cfg, "runtime", "enable_direct_keyword_trigger", False):
            return False
        if not _group_message_mentions_this_bot(event):
            return False

        text = _safe_event_text(event)
        if not _direct_reference_text_allows_image_edit(text):
            return False

        try:
            message_chain = event.get_messages()
        except Exception:
            message_chain = getattr(getattr(event, "message_obj", None), "message", [])
        return _message_chain_has_image_or_reply(message_chain)


class DirectTextImageGenerationFilter(filter.CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        if not _config_bool(cfg, "runtime", "enable_direct_keyword_trigger", False):
            return False
        if not _group_message_mentions_this_bot(event):
            return False

        text = _safe_event_text(event)
        # Let the current conversation resolve who a selfie/persona refers to.
        if _text_generation_needs_dialog_context(text):
            return False
        try:
            message_chain = event.get_messages()
        except Exception:
            message_chain = getattr(getattr(event, "message_obj", None), "message", [])
        has_image_or_reply = _message_chain_has_image_or_reply(message_chain)

        if _description_reference_text_allows_image_generation(text):
            return has_image_or_reply

        if has_image_or_reply and _direct_reference_text_allows_image_edit(text):
            return False

        return _direct_text_allows_image_generation(text)


OPENAI_SIZE_TO_RATIO = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "2560x1440": "16:9",
    "1440x2560": "9:16",
    "2880x2880": "1:1",
    "3840x2160": "16:9",
    "2160x3840": "9:16",
}

# Keyword sets used by ``ImageAPIError.should_try_other_profile`` to decide
# whether retrying once with the other compat profile is worth attempting.
#
# A profile swap helps when the upstream rejected our payload format. It does
# NOT help and would waste a request when the failure is auth, quota,
# moderation, or a missing model. The lists below are kept conservative so
# generic words (``policy`` alone, ``quota`` alone) do not accidentally swallow
# legitimate parameter errors that mention them in passing.
CONTENT_POLICY_KEYWORDS = (
    "content_policy",
    "content policy",
    "moderation",
    "safety system",
    "rejected by safety",
    "violates",
    "violate",
    "inappropriate",
)
SAFETY_GUARDRAIL_ERROR_KEYWORDS = (
    "guardrails around nudity",
    "nudity, sexuality, or erotic content",
    "may violate our guardrails",
    "erotic content",
    "sexuality",
    "nudity",
    "content_policy",
    "content policy",
    "safety policy",
    "policy violation",
    "rejected by safety",
)
SAFETY_DIALOG_REPLY_PROMPT_TEMPLATE = """
你是群聊里的 bot。刚才一次{operation}请求被上游图像接口的安全规则拦截了。

请直接回复用户一条自然、简短、中文的消息，要求：
1. 说明这次请求可能被判断为包含裸露、性暗示或成人向内容，所以没有生成/修改图片。
2. 不要输出英文错误、HTTP 状态码、接口名或技术细节。
3. 不要试图绕过安全规则，不要提供规避审查的办法。
4. 建议用户改成明确的非成人、非裸露描述后再试，可以提示从服装、材质、构图、风格、动作等安全维度重新描述。
5. 只输出最终回复，不要 Markdown，不要代码块。
""".strip()
BILLING_KEYWORDS = (
    "insufficient_quota",
    "insufficient quota",
    "billing",
    "exceeded your current quota",
    "exceeded your quota",
    "credit",
    "balance",
    "余额",
)
NON_PROFILE_ERROR_KEYWORDS = (
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "forbidden",
    "model not found",
    "model_not_found",
    "unknown model",
    "does not exist",
)

DEFAULT_API_TIMEOUT = 300

# aiohttp's default User-Agent looks like ``Python/3.x aiohttp/3.y``. Some
# middlemen place broad Cloudflare rules in front of API endpoints that block
# that generic UA before the request reaches the model server. Identify this
# plugin as a non-browser API client by default, while still allowing users to
# override the header with ``api.user_agent`` for stricter deployments.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; AstrBotGPTImagePlugin/1.0; "
    "+https://github.com/starmiaoa/astrbot-plugin-gpt-image)"
)

# Pattern that identifies Cloudflare anti-bot / browser-challenge / managed
# challenge HTML responses (typically returned with HTTP 403 and a ``cf-ray``
# header). Used to surface a clearer error than raw HTML to the user.
#
# CF Ray IDs are exactly 16 hex chars, optionally followed by an upper-case
# datacenter code like ``-LAX``. Anchor the search to ``cf-ray`` / ``Ray ID`` so
# we do not accidentally report an unrelated asset hash from the challenge page.
_CF_RAY_PATTERN = re.compile(
    r"(?:cf[-_ ]?ray|ray\s+id)[^0-9a-f]{0,120}([0-9a-f]{16}(?:-[A-Z]{2,4})?)",
    re.IGNORECASE,
)


class ImageAPIError(RuntimeError):
    """HTTP error returned from the image API.

    Wraps the status, parsed message, and (truncated) raw body so the
    caller can decide whether the failure is a parameter format issue
    that warrants a retry with the other compat profile.
    """

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status = status
        self.message = message
        self.body = body

    def should_try_other_profile(self) -> bool:
        """Whether retrying once with the other compat profile is useful.

        Many OpenAI-compatible proxies return vague 5xx errors such as
        ``openai_error`` for payloads they cannot translate. We only use this
        on the first profile attempt, so a permissive answer costs at most one
        extra request before surfacing the real error.
        """
        if self.status in {401, 403, 404, 429}:
            return False
        haystack = f"{self.message}\n{self.body}".lower()
        if any(keyword in haystack for keyword in CONTENT_POLICY_KEYWORDS):
            return False
        if any(keyword in haystack for keyword in BILLING_KEYWORDS):
            return False
        if any(keyword in haystack for keyword in NON_PROFILE_ERROR_KEYWORDS):
            return False
        return self.status in {400, 422}


DEFAULT_TOOL_GUIDE = """
# GPT Image 画图工具使用说明

你可以调用 `generate_image_with_gpt_image_2` 工具生成或修改图片。这个工具会真的提交图片生成任务 普通文字问答不要调用它。

什么时候调用：
- 群聊里只有用户明确 @ 你时才可以调用；未 @ 你的群消息即使包含图片、改图、重绘、生图等关键词，也不要调用。
- 只有用户明确要求你现在输出一张图片时才调用 例如“画一张...”“生成一张图...”“出图...”“做一张海报/头像/壁纸/图标...”；“再生成一遍”也算明确出图。
- 只有用户明确要求修改或参考图片 且当前消息或明确引用消息里有图片时才设置 `use_reference_images=true` 例如“改这张图...”“把图里的...换成...”“参考这张图生成...”；“改成动漫画风”“重画成手办材质”也算明确改图。
- `use_reference_images=false` 就是纯文生图，不会因“改图”等关键词自动带图。之前未明确引用的聊天图片不会被自动沿用；要使用旧图，请用户回复/引用那张图或重新附图。
- 如果用户想改图但没有发图或引用图 不要调用工具 先请用户发送或引用图片。
- 有参考图或引用图时，结合用户要求判断是改图、重绘、参考图生成，还是按图中文字/描述生成，并选择对应参数。
- 参考图不需要你完整复述进 `prompt`；`prompt` 写用户要改什么/生成什么即可。仅在 `use_reference_images=true` 时，插件收集本条消息和明确引用消息里的图片，并交给后续提示词增强模型读取参考图细节。
- 如果用户只是讨论图片 询问建议 写提示词 改提示词 翻译提示词 分析图片风格 规划设计方案 询问参数或问能不能生成 不要调用工具 直接用文字回答。
- 对话模型负责识别明确的生图/改图意图，并结合本轮对话选定画面主体、组织完整视觉提示词。插件负责提交任务，不会替无参考图的工具请求重新选择主体或套用固定人设。
- 用户表达不明确时不要主动生成 先用文字确认是否需要现在出图。
- 同一条用户消息最多调用一次工具 先生成一张图 除非用户后续继续要求修改或再生成 不要在同一轮对话里重复调用

参数怎么传：
- 用户明确要求原生 GIF 动图时，设置 `output_format="gif"`；`frame_count` 为 2–16 帧，`frame_duration_ms` 为每帧 20–2000 毫秒（越小越快，按 10 毫秒规整）。未指定时使用配置默认 9 帧、200 毫秒；不要自行增加帧数。GIF 的 `size` 可为 `auto` 或宽高均为 16 的倍数，例如 `512x512`；不要传 `aspect_ratio` 或 `resolution`，它们只用于静态图。本插件当前 GIF 入口仅支持纯文字生成，不支持参考图或改图；遇到这种请求请说明插件限制，不要丢弃参考图偷偷改为纯文字生成。未要求动图时不要设置 gif。
- `prompt` 写完整、具体、可独立理解的视觉需求。改图时说明用户要保留什么、改变什么、参考图之间的关系。
- 主体由你根据用户本轮要求和对话语境选定。只有用户明确要求画当前 bot 的“你、你的自拍、你的形象、本体”等形象时，才从本轮实际生效的人格组织可见外观、服装、表情、姿态、场景和风格，并完整写入 `prompt`。不要只写“你”或人格名称。
- 用户指定其他人物或角色时，包括其他人的自拍，必须保留指定主体，不得因为“自拍、形象、人设”等关键词套用 bot 的形象。无参考图时插件直接使用这份 `prompt`，仅追加显式 `style`，不再做人设改写、提示词增强、英文化或统一前后缀拼接。
- `resolution` 只用于 `1k`/`2k` 清晰度档位；当前上游实测 `resolution=4k` 不会稳定返回真实 4K，所以用户要 4K 时请改用 `size`。
- `aspect_ratio` 是画面比例：文生图时请根据用途主动选择最合适比例，例如头像/图标 `1:1`，海报/手机壁纸 `9:16`，电脑壁纸/横版海报 `16:9`，产品横幅 `21:9`，普通横图 `3:2`，普通竖图 `2:3`；如果用户明确要求比例就按用户要求传。改图时除非用户明确要求新比例，否则留空，插件会尽量按第一张参考图比例。
- `size` 是精确像素尺寸。可用档位：`1024x1024`、`1536x1024`、`1024x1536`、`2560x1440`、`1440x2560`、`2880x2880`、`3840x2160`、`2160x3840`。用户要 4K 横图用 `3840x2160`，4K 竖图用 `2160x3840`。
- 为了提高成功率并降低上游负载，日常不要主动传 `resolution=2k` 或 `quality=high`；尤其有参考图/引用图的改图请求，除非用户明确写了 2K、4K、高清、高质量、精修、最终稿或具体像素尺寸，否则 `resolution`、`quality` 都留空或 `auto`。
- `quality` 日常留空或 `auto`；只有用户明确要求精细、最终稿、高清时才用 `high`。
- `transparent_background=true` 用于 logo、贴纸、图标、UI 素材、抠图、透明底。
- `style` 可放额外风格词，比如写实、动漫、水彩、产品渲染、扁平图标。
- GPT 图像编辑最多会读取 16 张参考图；插件会自动从当前消息和引用消息里收集。

工具返回状态后 只按你当前的人格和语气回复一句确认 让用户知道这次图像任务已经接下并开始处理。回复末尾必须带出工具返回的流程ID原文，并提示“复杂需求可以用 /plan 进入高级画图模式”。不要补充接口、模型、耗时等技术细节，不要说已经完成，不要再次调用同一个工具。
""".strip()

DEFAULT_PROMPT_ENHANCE_TEMPLATE = """
你是图像生成提示词工程师。请根据用户需求和参考图片，输出一段适合 GPT Image 图像编辑/生成接口使用的最终提示词。

用户原始需求：
{user_prompt}

工具传入的额外风格参数：
{style}

参考图片数量：
{image_count}

请遵守：
1. 先准确理解参考图主体、构图、画面关系、颜色、材质、表情、动作和重要细节。
2. 如果参考图中有文字，请逐字列出文字内容，并说明大概位置；用户要求保留文字时，必须在最终提示词里强调文字清晰可读、内容和位置尽量保持一致。
3. 如果用户要求重绘、转画风或改图，请明确哪些元素需要保留，哪些元素需要改变。
4. 把用户要求转写成完整、具体、可独立理解的图像提示词，适合直接提交给图片生成模型。
5. 不要新增用户没有要求的主体、文字、logo、水印或剧情。
6. 只输出最终提示词，不要解释，不要 Markdown，不要代码块。
""".strip()

DEFAULT_PROMPT_SUBJECT_PLACEHOLDER = "[在此处替换为您想要生成的主体内容]"

DEFAULT_PROMPT_TRANSLATION_TEMPLATE = """
You are an image prompt localization and rewriting assistant.

Task:
Rewrite the following final image prompt into clear, natural English suitable for an image generation model.

Final image prompt:
{prompt}

Requirements:
1. Output only the final English image prompt. Do not explain, do not use Markdown, and do not wrap it in quotes.
2. Preserve every concrete visual requirement, composition constraint, style, subject identity, clothing, expression, pose, background, lighting, and negative constraint.
3. If the prompt asks the image to contain specific visible text, keep that visible text exactly in its original language and make it clear that the text should appear in the image.
4. Do not add new characters, logos, watermarks, captions, or story elements.
5. Keep safety and modesty constraints explicit if they are present.
""".strip()


DEFAULT_PLAN_PROMPT_TEMPLATE = """
你是 GPT Image · 龙绘 的高级画图规划助手。用户正在使用 /plan 模式，此阶段只整理图像需求，不要生成图片，不要拒绝，不要输出闲聊。

用户累计需求：
{raw_prompt}

用户本次追加：
{latest_prompt}

已有整理稿：
{previous_plan}

当前参数：
{options}

参考图数量：{reference_count}

请根据用户需求和参考图，把需求整理成一份可确认的图像计划。若用户本次追加是在修改上一版计划，请合并修改后输出新版计划。

输出必须使用下面四个小节，标题保持原样：
【图像需求】
用中文概括主体、动作/姿态、服装/外观、背景、风格、构图、光影、必须保留/改变的元素。

【参考图使用方式】
说明每张参考图大致承担什么作用；如果没有参考图，写“无参考图”。

【最终提示词】
写一段可以直接提交给 GPT Image 图像生成/编辑接口的最终中文提示词。只写画面要求，不要写解释、流程、道歉或确认话术。

【可修改项】
列出还可以继续调整的画幅、风格、细节、参考图关系或质量参数。
""".strip()


@register(
    PLUGIN_ID,
    "starmiaoa",
    "GPT Image · 龙绘：自然对话生图、改图、流程追踪与 /plan 高级画图",
    "1.4.1",
)
class GPTImage2Plugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        # Keep legacy keys in the schema until AstrBot has loaded and migrated
        # them. Its integrity check otherwise removes old keys before __init__.
        if self.config.get("_config_version", 0) == 0:
            old_image = self.config.get("image", {})
            gif = self.config.setdefault("gif", {})
            for old_key, new_key, default in (
                ("gif_frame_count", "frame_count", 9),
                ("gif_frame_duration_ms", "frame_duration_ms", 200),
            ):
                if old_key in old_image and gif.get(new_key, default) == default:
                    gif[new_key] = old_image[old_key]
            if old_image.get("output_format") == "gif" and gif.get("size", "auto") == "auto":
                gif["size"] = old_image.get("size", "auto")
            # The new opt-in switch remains off for old and new installations.
            # Never infer official API access from a model name or hostname.
            self.config["_config_version"] = 1
            if callable(getattr(self.config, "save_config", None)):
                self.config.save_config()
        self._data_dir = StarTools.get_data_dir()
        self._image_dir = self._data_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._trace_dir = self._data_dir / "traces"
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        max_concurrent = self._int_cfg("runtime", "max_concurrent", 1)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_tool_tasks: dict[str, asyncio.Task[tuple[str, str | None]]] = {}
        self._inflight_tool_turns: dict[str, asyncio.Task[tuple[str, str | None]]] = {}
        self._inflight_tool_traces: dict[str, dict[str, Any]] = {}
        self._inflight_tool_turn_traces: dict[str, dict[str, Any]] = {}
        self._plan_sessions: dict[str, dict[str, Any]] = {}
        self._plan_handled_events: dict[str, float] = {}
        # In-memory cache of the compat profile that succeeded last time for a
        # given (base_url, model, operation). Lets the next request skip the
        # auto-detect heuristic and the failure-retry roundtrip.
        self._compat_profile_cache: dict[tuple[str, str, str], str] = {}

    def _group_message_mentions_this_bot(self, event: AstrMessageEvent) -> bool:
        return _group_message_mentions_this_bot(event)

    @filter.command("生图")
    async def generate_command(self, event: AstrMessageEvent):
        if not self._group_message_mentions_this_bot(event):
            logger.info("GPT Image /生图 ignored in group because the message did not mention this bot")
            event.stop_event()
            return

        prompt = self._extract_command_prompt(event, ("生图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：/生图 一只赛博猫坐在霓虹窗边 --size 1024x1024 --quality high"
            ).stop_event()
            return

        image_prompt = prompt
        trace = self._new_image_trace(
            event,
            trigger="/生图",
            operation="generation",
            raw_user_prompt=prompt,
            prompt=image_prompt,
            reference_image_paths=[],
            options=opts,
        )
        self._trace_route_decision(
            trace,
            route="text_to_image",
            reason="explicit_generate_command",
            operation="generation",
            reference_image_paths=[],
        )
        task = asyncio.create_task(
            self._generate_image(
                prompt=image_prompt,
                raw_user_prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                output_format=opts.get("output_format"),
                frame_count=opts.get("frame_count"),
                frame_duration_ms=opts.get("frame_duration_ms"),
                trace=trace,
            )
        )
        self._schedule_background_delivery(event, task, error_prefix="图片生成失败", trace=trace)
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message(flow_id=self._trace_flow_id(trace)))

    @filter.command("改图")
    async def edit_command(self, event: AstrMessageEvent):
        if not self._group_message_mentions_this_bot(event):
            logger.info("GPT Image /改图 ignored in group because the message did not mention this bot")
            event.stop_event()
            return

        prompt = self._extract_command_prompt(event, ("改图",))
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：发送或引用图片后，输入 /改图 把背景换成夜晚城市 --size 1536x1024 --quality high"
            ).stop_event()
            return

        reference_source_records: list[dict[str, str]] = []
        reference_paths = await self._collect_reference_image_paths(
            event, stabilize=True, source_records=reference_source_records,
        )
        if not reference_paths:
            yield event.plain_result("没有找到可用参考图。请在同一条消息里发图，或回复/引用一张图后再使用 /改图。").stop_event()
            return

        trace = self._new_image_trace(
            event,
            trigger="/改图",
            operation="edit",
            raw_user_prompt=prompt,
            prompt=prompt,
            reference_image_paths=reference_paths,
            options=opts,
        )
        self._trace_route_decision(
            trace,
            route="image_edit",
            reason="explicit_edit_command",
            operation="edit",
            reference_image_paths=reference_paths,
        )
        self._trace_stage(
            trace, "reference_selection", status="succeeded",
            policy="current_or_explicit_quote_only", sources=reference_source_records,
            selected_paths=reference_paths,
        )
        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                raw_user_prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                output_format=opts.get("output_format"),
                frame_count=opts.get("frame_count"),
                frame_duration_ms=opts.get("frame_duration_ms"),
                reference_image_paths=reference_paths,
                trace=trace,
            )
        )
        self._schedule_background_delivery(event, task, error_prefix="图片修改失败", trace=trace)
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message(reference_count=len(reference_paths), flow_id=self._trace_flow_id(trace)))

    @filter.command("绘图")
    async def native_draw_command(self, event: AstrMessageEvent):
        prompt = self._extract_command_prompt(event, ("绘图",)).strip()
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            yield event.plain_result(
                "用法：/绘图 提示词（可同消息发送或引用参考图）。这个命令会原样提交提示词，不做增强、翻译或压缩。"
            ).stop_event()
            return

        reference_source_records: list[dict[str, str]] = []
        reference_paths = await self._collect_reference_image_paths(
            event,
            include_recent=False,
            stabilize=True,
            source_records=reference_source_records,
        )
        operation = "edit" if reference_paths else "generation"
        trace = self._new_image_trace(
            event,
            trigger="/绘图",
            operation=operation,
            raw_user_prompt=prompt,
            prompt=prompt,
            reference_image_paths=reference_paths,
            options={**opts, "native_passthrough": True},
        )
        self._trace_route_decision(
            trace,
            route="native_image_edit" if reference_paths else "native_text_to_image",
            reason="explicit_native_draw_command",
            operation=operation,
            reference_image_paths=reference_paths,
            native_passthrough=True,
        )
        self._trace_stage(
            trace, "reference_selection", status="succeeded",
            policy="current_or_explicit_quote_only", sources=reference_source_records,
            selected_paths=reference_paths,
        )
        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                raw_user_prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                output_format=opts.get("output_format"),
                frame_count=opts.get("frame_count"),
                frame_duration_ms=opts.get("frame_duration_ms"),
                reference_image_paths=reference_paths,
                trace=trace,
                skip_prompt_enhance=True,
                skip_prompt_translation=True,
                raw_prompt_passthrough=True,
            )
        )
        self._schedule_background_delivery(
            event,
            task,
            error_prefix="图片修改失败" if reference_paths else "图片生成失败",
            trace=trace,
        )
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message(flow_id=self._trace_flow_id(trace)))

    @filter.custom_filter(DirectTextImageGenerationFilter)
    async def direct_text_image_generation_request(self, event: AstrMessageEvent):
        prompt = self._event_text(event)
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            return
        # Do not intercept requests whose subject needs conversation context.
        if _text_generation_needs_dialog_context(prompt):
            return
        try:
            render_options = self._resolve_render_options(
                output_format=opts.get("output_format"), frame_count=opts.get("frame_count"),
                frame_duration_ms=opts.get("frame_duration_ms"), size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"), resolution=opts.get("resolution"),
            )
        except Exception as exc:
            event.stop_event()
            yield event.plain_result(f"图片参数不可用：{self._friendly_error(exc)}")
            return
        description_reference_paths: list[str] = []
        description_reference_texts: list[str] = []
        if _description_reference_text_allows_image_generation(prompt):
            description_reference_paths = await self._collect_reference_image_paths(event)
            description_reference_texts = self._collect_reference_description_texts(event)

        preprocess_stage: dict[str, Any] | None = None
        if render_options["output_format"] == "gif" and description_reference_paths:
            event.stop_event()
            yield event.plain_result("本插件的 GIF 动图目前仅支持纯文字生成，不支持参考图；本次不会丢弃参考图或调用描述模型。")
            return
        if description_reference_paths or description_reference_texts:
            try:
                image_prompt = await self._enhance_prompt_from_description_reference(
                    prompt=prompt,
                    reference_image_paths=description_reference_paths,
                    reference_texts=description_reference_texts,
                )
                preprocess_stage = {
                    "name": "description_reference_prompt",
                    "status": "succeeded",
                    "input_prompt": prompt,
                    "output_prompt": image_prompt,
                    "description_reference_images": description_reference_paths,
                    "description_reference_texts": description_reference_texts,
                }
            except Exception as exc:
                logger.warning(
                    "GPT Image description reference prompt enhancement failed: %s",
                    self._friendly_error(exc),
                    exc_info=True,
                )
                event.stop_event()
                yield event.plain_result(f"读取图片/引用里的描述失败：{self._friendly_error(exc)}")
                return
        elif _description_reference_text_allows_image_generation(prompt):
            event.stop_event()
            yield event.plain_result("没有找到可用的引用图片或引用文字。请引用包含描述文字的图片后再说“按这个描述生成一张图”。")
            return
        else:
            image_prompt = prompt

        turn_key = self._tool_turn_key(event)
        turn_task = self._inflight_tool_turns.get(turn_key)
        if turn_task and not turn_task.done():
            trace = self._inflight_tool_turn_traces.get(turn_key)
            event.stop_event()
            yield event.plain_result(self._tool_submitted_message(flow_id=self._trace_flow_id(trace)))
            return
        if turn_task and turn_task.done():
            self._inflight_tool_turns.pop(turn_key, None)
            self._inflight_tool_turn_traces.pop(turn_key, None)

        request_key = self._tool_request_key(
            event,
            prompt=image_prompt,
            size=opts.get("size"),
            aspect_ratio=opts.get("aspect_ratio"),
            resolution=opts.get("resolution"),
            quality=opts.get("quality"),
            style=opts.get("style"),
            transparent_background=opts.get("transparent_background", False),
            reference_paths=[],
            render_options=render_options,
        )
        existing_task = self._inflight_tool_tasks.get(request_key)
        if existing_task and not existing_task.done():
            trace = self._inflight_tool_traces.get(request_key)
            event.stop_event()
            yield event.plain_result(self._tool_submitted_message(flow_id=self._trace_flow_id(trace)))
            return
        if existing_task and existing_task.done():
            self._inflight_tool_tasks.pop(request_key, None)
            self._inflight_tool_traces.pop(request_key, None)

        trace = self._new_image_trace(
            event,
            trigger="direct_text",
            operation="generation",
            raw_user_prompt=prompt,
            prompt=image_prompt,
            reference_image_paths=[],
            options=opts,
        )
        route = "text_to_image"
        reason = "direct_text_generation"
        if description_reference_paths or description_reference_texts:
            route = "image_description_generation"
            reason = "description_reference_context"
        self._trace_route_decision(
            trace,
            route=route,
            reason=reason,
            operation="generation",
            reference_image_paths=[],
            description_reference_image_count=len(description_reference_paths),
            description_reference_text_count=len(description_reference_texts),
        )
        if preprocess_stage:
            name = str(preprocess_stage.pop("name", "preprocess_prompt"))
            self._trace_stage(trace, name, **preprocess_stage)
        task = asyncio.create_task(
            self._generate_image(
                prompt=image_prompt,
                raw_user_prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                render_options=render_options,
                trace=trace,
            )
        )
        self._inflight_tool_tasks[request_key] = task
        self._inflight_tool_turns[turn_key] = task
        self._inflight_tool_traces[request_key] = trace
        self._inflight_tool_turn_traces[turn_key] = trace

        def _clear_inflight(done_task: asyncio.Task[tuple[str, str | None]]) -> None:
            if self._inflight_tool_tasks.get(request_key) is done_task:
                self._inflight_tool_tasks.pop(request_key, None)
                self._inflight_tool_traces.pop(request_key, None)
            if self._inflight_tool_turns.get(turn_key) is done_task:
                self._inflight_tool_turns.pop(turn_key, None)
                self._inflight_tool_turn_traces.pop(turn_key, None)

        task.add_done_callback(_clear_inflight)
        logger.info(
            "GPT Image direct text generation intercepted before LLM prompt_len=%d final_prompt_len=%d",
            len(prompt),
            len(image_prompt),
        )
        self._schedule_background_delivery(event, task, error_prefix="图片生成失败", trace=trace)
        event.stop_event()
        yield event.plain_result(self._tool_submitted_message(flow_id=self._trace_flow_id(trace)))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=9600)
    async def image2_plan_prefix_guard(self, event: AstrMessageEvent):
        raw_text = self._event_text(event)
        payload = _plan_command_payload(raw_text)
        if payload is None:
            return

        if raw_text.lower().startswith("plan"):
            try:
                message_chain = event.get_messages()
            except Exception:
                message_chain = getattr(getattr(event, "message_obj", None), "message", [])
            if not _message_chain_has_image_or_reply(message_chain):
                return

        logger.info("GPT Image · 龙绘 Plan prefix guard intercepted plan-like message")
        async for result in self._handle_image2_plan(event, payload):
            yield result

    @filter.custom_filter(DirectReferenceImageEditFilter)
    async def direct_reference_image_edit_request(self, event: AstrMessageEvent):
        prompt = self._event_text(event)
        if _text_requests_gif(prompt):
            return
        opts, prompt = self._parse_inline_options(prompt)
        if not prompt:
            return

        try:
            render_options = self._resolve_render_options(
                output_format=opts.get("output_format"), frame_count=opts.get("frame_count"),
                frame_duration_ms=opts.get("frame_duration_ms"), size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"), resolution=opts.get("resolution"),
                reference_image_paths=["requested-reference"],
            )
        except Exception as exc:
            event.stop_event()
            yield event.plain_result(f"图片参数不可用：{self._friendly_error(exc)}")
            return

        reference_source_records: list[dict[str, str]] = []
        reference_paths = await self._collect_reference_image_paths(
            event, stabilize=True, source_records=reference_source_records,
        )
        if not reference_paths:
            event.stop_event()
            yield event.plain_result(
                "没有找到可用参考图。请在同一条消息里发图，或回复/引用一张图后再说明要怎么改。"
            )
            return

        turn_key = self._tool_turn_key(event)
        turn_task = self._inflight_tool_turns.get(turn_key)
        if turn_task and not turn_task.done():
            trace = self._inflight_tool_turn_traces.get(turn_key)
            event.stop_event()
            yield event.plain_result(
                self._tool_submitted_message(
                    reference_count=len(reference_paths),
                    flow_id=self._trace_flow_id(trace),
                )
            )
            return
        if turn_task and turn_task.done():
            self._inflight_tool_turns.pop(turn_key, None)
            self._inflight_tool_turn_traces.pop(turn_key, None)

        request_key = self._tool_request_key(
            event,
            prompt=prompt,
            size=opts.get("size"),
            aspect_ratio=opts.get("aspect_ratio"),
            resolution=opts.get("resolution"),
            quality=opts.get("quality"),
            style=opts.get("style"),
            transparent_background=opts.get("transparent_background", False),
            reference_paths=reference_paths,
            render_options=render_options,
        )
        existing_task = self._inflight_tool_tasks.get(request_key)
        if existing_task and not existing_task.done():
            trace = self._inflight_tool_traces.get(request_key)
            event.stop_event()
            yield event.plain_result(
                self._tool_submitted_message(
                    reference_count=len(reference_paths),
                    flow_id=self._trace_flow_id(trace),
                )
            )
            return
        if existing_task and existing_task.done():
            self._inflight_tool_tasks.pop(request_key, None)
            self._inflight_tool_traces.pop(request_key, None)

        trace = self._new_image_trace(
            event,
            trigger="direct_reference_edit",
            operation="edit",
            raw_user_prompt=prompt,
            prompt=prompt,
            reference_image_paths=reference_paths,
            options=opts,
        )
        self._trace_route_decision(
            trace,
            route="image_edit",
            reason="reference_context_filter",
            operation="edit",
            reference_image_paths=reference_paths,
        )
        self._trace_stage(
            trace, "reference_selection", status="succeeded",
            policy="current_or_explicit_quote_only", sources=reference_source_records,
            selected_paths=reference_paths,
        )
        task = asyncio.create_task(
            self._generate_image(
                prompt=prompt,
                raw_user_prompt=prompt,
                size=opts.get("size"),
                aspect_ratio=opts.get("aspect_ratio"),
                resolution=opts.get("resolution"),
                quality=opts.get("quality"),
                style=opts.get("style"),
                transparent_background=opts.get("transparent_background", False),
                reference_image_paths=reference_paths,
                render_options=render_options,
                trace=trace,
            )
        )
        self._inflight_tool_tasks[request_key] = task
        self._inflight_tool_turns[turn_key] = task
        self._inflight_tool_traces[request_key] = trace
        self._inflight_tool_turn_traces[turn_key] = trace

        def _clear_inflight(done_task: asyncio.Task[tuple[str, str | None]]) -> None:
            if self._inflight_tool_tasks.get(request_key) is done_task:
                self._inflight_tool_tasks.pop(request_key, None)
                self._inflight_tool_traces.pop(request_key, None)
            if self._inflight_tool_turns.get(turn_key) is done_task:
                self._inflight_tool_turns.pop(turn_key, None)
                self._inflight_tool_turn_traces.pop(turn_key, None)

        task.add_done_callback(_clear_inflight)
        logger.info(
            "GPT Image direct reference edit intercepted before LLM refs=%d prompt_len=%d",
            len(reference_paths),
            len(prompt),
        )
        self._schedule_background_delivery(event, task, error_prefix="图片修改失败", trace=trace)
        event.stop_event()
        yield event.plain_result(
            self._tool_submitted_message(
                reference_count=len(reference_paths),
                flow_id=self._trace_flow_id(trace),
            )
        )

    @filter.command("生图帮助")
    async def gptimage_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "GPT Image · 龙绘 已加载。\n"
            "命令：/生图 提示词；/改图 提示词（同消息发图，或回复/引用图片）；/绘图 提示词（原样直通，不增强不翻译）。\n"
            "高级模式：/plan 进入多轮确认式画图；/plan confirm 确认生成；/plan quit 退出。\n"
            "参数：--resolution 1k|2k，--ratio 1:1|16:9|9:16|3:2|2:3，"
            "--size 1024x1024|1536x1024|1024x1536|2560x1440|1440x2560|2880x2880|3840x2160|2160x3840|auto，"
            "--quality auto|low|medium|high，--style 风格，--transparent。\n"
            "GIF 动图：--format gif --frames 9 --frame-ms 200 --size 512x512；2–16 帧，每帧 20–2000 毫秒。本插件目前仅支持纯文字 GIF，不支持参考图、--ratio 或 --resolution。需供应商已开放原生 GIF。\n"
            "说明：像素尺寸用 --size，清晰度档位用 --resolution，比例用 --ratio。\n"
            f"多图参考：最多 {MAX_GPT_IMAGE_REFERENCE_IMAGES} 张，只使用本条消息或明确引用的图片；修改之前的图片请回复/引用它。\n"
            "查询：/生图流程 流程ID，优先查看实际提交提示词和结果；仅原请求者在原会话可查，管理员跨会话排查请私聊。"
        ).stop_event()

    @filter.command("plan")
    async def image2_plan(self, event: AstrMessageEvent):
        text = self._extract_command_prompt(event, ("plan",)).strip()
        async for result in self._handle_image2_plan(event, text):
            yield result

    async def _handle_image2_plan(self, event: AstrMessageEvent, text: str):
        if not self._mark_plan_event(event):
            event.stop_event()
            return

        text = _strip_image_placeholders(text).strip()
        normalized = _normalized_request_text(text)
        key = self._plan_session_key(event)
        expired_current = self._cleanup_expired_plan_sessions(current_key=key)

        if normalized in {"quit", "cancel", "exit", "退出", "取消", "结束"}:
            existed = self._plan_sessions.pop(key, None)
            event.stop_event()
            yield event.plain_result("已退出 GPT Image · 龙绘 Plan 模式。" if existed else "当前没有正在进行的 Plan。")
            return

        if normalized in {"status", "状态", "查看"}:
            session = self._plan_sessions.get(key)
            event.stop_event()
            if expired_current:
                yield event.plain_result(self._plan_expired_text())
                return
            if session and session.get("planning"):
                yield event.plain_result("当前计划正在整理，请稍候再查看状态。")
                return
            if session and str(session.get("planned_response") or "").strip():
                try:
                    await self._prepare_plan_submission(session)
                except Exception as exc:
                    yield event.plain_result(f"最终提示词准备失败，尚未提交生图；请用 /plan status 查看或重试：{self._friendly_error(exc)}")
                    return
                yield event.chain_result(self._build_plan_session_chain(event, session))
            else:
                yield event.plain_result(self._format_plan_session_status(session) if session else self._plan_help_text())
            return

        if normalized in {"confirm", "ok", "go", "确认", "确认生成", "开始生成"}:
            session = self._plan_sessions.get(key)
            if expired_current:
                event.stop_event()
                yield event.plain_result(self._plan_expired_text())
                return
            if not session:
                event.stop_event()
                yield event.plain_result("当前没有可确认的 Plan。先用 /plan 你的画图需求 建一个计划。")
                return
            raw_prompt = self._plan_raw_prompt(session)
            if not raw_prompt:
                event.stop_event()
                yield event.plain_result("这个 Plan 还没有画图需求。用 /plan 补充你想画什么，再 /plan confirm。")
                return

            if session.get("confirming") or session.get("planning"):
                event.stop_event()
                yield event.plain_result("当前计划正在准备，请稍候再查看状态；不会重复提交生图。")
                return

            opts = dict(session.get("options") or {})
            planned_prompt = self._plan_final_prompt(session) or raw_prompt
            session["confirming"] = True
            try:
                expected_fingerprint = self._plan_submission_fingerprint(session)
                reference_paths = self._validated_plan_reference_paths(session)
                prepared = await self._prepare_plan_submission(session)
                if (
                    self._plan_sessions.get(key) is not session
                    or self._cleanup_expired_plan_sessions(current_key=key)
                    or session.get("planning")
                    or self._plan_submission_fingerprint(session) != expected_fingerprint
                ):
                    raise RuntimeError("准备期间计划已修改、退出或过期，请查看最新计划后重新确认")
                reference_paths = self._validated_plan_reference_paths(session)
            except Exception as exc:
                event.stop_event()
                yield event.plain_result(f"最终提示词准备失败，尚未提交生图；请查看当前计划后重试 /plan confirm：{self._friendly_error(exc)}")
                return
            finally:
                session.pop("confirming", None)
            cn_prompt = prepared["chinese_prompt"]
            mixed_prompt = prepared["final_prompt"]
            trace = self._new_image_trace(
                event,
                trigger="/plan confirm",
                operation="edit" if reference_paths else "generation",
                raw_user_prompt=raw_prompt,
                prompt=planned_prompt,
                reference_image_paths=reference_paths,
                options=opts,
            )
            self._trace_route_decision(
                trace,
                route="plan_edit" if reference_paths else "plan_generation",
                reason="plan_confirm",
                operation="edit" if reference_paths else "generation",
                reference_image_paths=reference_paths,
            )
            self._trace_stage(
                trace, "reference_selection", status="succeeded",
                policy="current_or_explicit_quote_only",
                sources=[record for record in session.get("reference_sources", []) if record.get("path") in reference_paths],
                selected_paths=reference_paths,
            )
            self._trace_stage(
                trace,
                "plan_confirm",
                status="succeeded",
                chinese_prompt=cn_prompt,
                mixed_prompt=mixed_prompt,
                planned_response=session.get("planned_response", ""),
                planned_prompt=planned_prompt,
                reference_count=len(reference_paths),
                options=opts,
            )
            task = asyncio.create_task(
                self._generate_image(
                    prompt=mixed_prompt,
                    raw_user_prompt=raw_prompt,
                    size=opts.get("size"),
                    aspect_ratio=opts.get("aspect_ratio"),
                    resolution=opts.get("resolution"),
                    quality=opts.get("quality"),
                    style=opts.get("style"),
                    transparent_background=opts.get("transparent_background", False),
                    reference_image_paths=reference_paths,
                    trace=trace,
                    skip_prompt_enhance=True,
                    skip_prompt_translation=True,
                    raw_prompt_passthrough=True,
                    advanced=True,
                    expected_image_selection=prepared["image_selection"],
                    render_options=prepared["render_options"],
                )
            )
            self._schedule_background_delivery(
                event,
                task,
                error_prefix="图片修改失败" if reference_paths else "图片生成失败",
                trace=trace,
            )
            self._plan_sessions.pop(key, None)
            event.stop_event()
            yield event.chain_result(
                self._build_plan_confirm_chain(
                    event,
                    trace=trace,
                    raw_prompt=raw_prompt,
                    cn_prompt=cn_prompt,
                    mixed_prompt=mixed_prompt,
                    options=opts,
                    reference_count=len(reference_paths),
                    owner_text=self._plan_owner_text(session, event),
                )
            )
            return

        opts, prompt = self._parse_inline_options(text)
        reference_source_records: list[dict[str, str]] = []
        reference_paths = await self._collect_reference_image_paths(
            event,
            include_recent=False,
            stabilize=True,
            source_records=reference_source_records,
        )
        session = self._plan_sessions.get(key)
        if not session:
            session = {
                "created_at": time.time(),
                "prompts": [],
                "options": {},
                "reference_image_paths": [],
                "owner": self._plan_owner_info(event),
            }
            self._plan_sessions[key] = session
        elif "owner" not in session:
            session["owner"] = self._plan_owner_info(event)

        if prompt:
            session.setdefault("prompts", []).append(prompt)
        session.pop("prepared_submission", None)
        self._merge_plan_options(session, opts)
        self._merge_plan_references(session, reference_paths)
        source_by_path = {record["path"]: record for record in session.get("reference_sources", []) if record.get("path")}
        source_by_path.update({record["path"]: record for record in reference_source_records if record.get("path")})
        session["reference_sources"] = [
            source_by_path[path] for path in session.get("reference_image_paths", []) if path in source_by_path
        ]
        session["updated_at"] = time.time()
        session["revision"] = int(session.get("revision", 0)) + 1
        revision = session["revision"]

        event.stop_event()
        if not self._plan_raw_prompt(session) and not session.get("reference_image_paths"):
            yield event.plain_result(self._plan_help_text())
            return
        try:
            self._freeze_plan_render_options(session)
        except Exception as exc:
            yield event.plain_result(f"Plan 参数不可用，尚未调用模型或提交生图：{self._friendly_error(exc)}")
            return
        session["planning"] = True
        try:
            await self._refresh_plan_session(session, latest_prompt=prompt)
        except Exception as exc:
            if session.get("revision") != revision or self._plan_sessions.get(key) is not session:
                yield event.plain_result("计划已更新或退出，本次旧整理结果已丢弃。")
                return
            fallback_prompt = self._plan_raw_prompt(session)
            session["planned_prompt"] = fallback_prompt
            session["planned_response"] = self._fallback_plan_response(session, error=self._friendly_error(exc))
            session["planning_error"] = self._friendly_error(exc)
            logger.warning("GPT Image · 龙绘 plan model failed; fallback to local plan: %s", self._friendly_error(exc), exc_info=True)
        finally:
            if session.get("revision") == revision:
                session.pop("planning", None)
        if self._plan_sessions.get(key) is not session:
            yield event.plain_result("计划已退出，本次整理结果已丢弃。")
            return
        try:
            await self._prepare_plan_submission(session)
        except Exception as exc:
            yield event.plain_result(f"最终提示词准备失败，尚未提交生图；请用 /plan status 查看或重试：{self._friendly_error(exc)}")
            return
        if self._plan_sessions.get(key) is not session:
            yield event.plain_result("计划已退出，本次整理结果已丢弃。")
            return
        if str(session.get("planned_response") or "").strip():
            yield event.chain_result(self._build_plan_session_chain(event, session))
        else:
            yield event.plain_result(self._format_plan_session_status(session))

    @filter.command("生图流程")
    async def gptimage_trace(self, event: AstrMessageEvent):
        flow_id = self._extract_command_prompt(event, ("生图流程",)).strip()
        if not flow_id:
            yield event.plain_result("用法：/生图流程 gi-20260518-123456-abcdef12").stop_event()
            return

        trace = self._load_image_trace(flow_id)
        if not trace or not self._can_access_image_trace(event, trace):
            yield event.plain_result("无法查询该流程。请由原请求者在原会话查询，或由 AstrBot 管理员私聊查询。").stop_event()
            return

        yield event.chain_result(self._build_trace_result_chain(trace)).stop_event()

    @filter.command("查生图")
    async def gptimage_trace_alias(self, event: AstrMessageEvent):
        flow_id = self._extract_command_prompt(event, ("查生图",)).strip()
        if not flow_id:
            yield event.plain_result("用法：/查生图 gi-20260518-123456-abcdef12").stop_event()
            return

        trace = self._load_image_trace(flow_id)
        if not trace or not self._can_access_image_trace(event, trace):
            yield event.plain_result("无法查询该流程。请由原请求者在原会话查询，或由 AstrBot 管理员私聊查询。").stop_event()
            return

        yield event.chain_result(self._build_trace_result_chain(trace)).stop_event()

    @llm_tool(name=TOOL_NAME)
    async def generate_image_with_gpt_image_2(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        quality: str = "",
        style: str = "",
        transparent_background: bool = False,
        use_reference_images: bool = False,
        output_format: str = "",
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
    ):
        """Generate or edit an image with a GPT image model and send it to the current chat.

        Use this tool only when the user explicitly asks you to generate or edit an
        image now. Set use_reference_images=true only when the user wants to edit,
        transform, combine, or follow images attached to the current message or explicitly
        quoted/replied message. Earlier unquoted chat images are never selected.
        Set false for text-only generation; edit-related words will not override false.
        Do not use it for normal text-only answers, prompt writing, design
        advice, image discussion, parameter questions, or ambiguous requests.
        Resolve the requested subject from the user's request and current conversation,
        then write a complete visual prompt. Only when the user explicitly asks for
        the current bot's selfie, avatar, body, or persona should you derive visible
        character details from the persona active in this turn. Do not pass only a
        persona name or a vague reference such as "yourself". Other people's selfies
        must keep those people as the subjects, never the bot's persona.
        Without reference images, the plugin submits your prompt without persona
        rewriting, enhancement, translation, or configured prefix/suffix templates;
        an explicit style parameter is appended deterministically. With reference
        images, the existing reference-image prompt enhancement remains available.

        Call this tool at most once for one user message. The tool sends the finished image directly to the user. After the tool reports
        background-task status, respond in your persona and naturally let the user know
        you accepted the image task and started working on it. End the reply with the exact flow ID
        returned by the tool and tell the user complex requests can use /plan for advanced multi-step image planning.
        Let length, phrasing, and punctuation follow your persona and context. Do not mention technical details or timing.

        Args:
            prompt(string): Required. A complete, concrete image prompt describing subject, style, composition, colors, lighting, text to include if any, and constraints from the user.
            size(string): Optional. For static images use auto, 1024x1024, 1536x1024, 1024x1536, 2560x1440, 1440x2560, 2880x2880, 3840x2160, or 2160x3840. GIF supports auto or positive width and height divisible by 16, such as 256x256 or 512x512, without static-image size remapping.
            aspect_ratio(string): Optional for static images. Desired aspect ratio, such as 1:1, 16:9, 9:16, 3:2, 2:3, 21:9. For static text-to-image, choose the best ratio for the intended use. For edits, leave empty unless the user asks to change it. For GIF always leave empty or auto; use size instead.
            resolution(string): Optional for static images: auto, 1k, or 2k. Do not use resolution=4k; use size=3840x2160 or size=2160x3840. For GIF always leave empty or auto; use size instead.
            quality(string): Optional. Image quality: auto, low, medium, or high. Use high only when the user asks for detailed/final artwork.
            style(string): Optional. Extra visual style words such as realistic, anime, watercolor, product render, poster, pixel art, or flat icon.
            transparent_background(boolean): Optional. Set true for logos, stickers, icons, UI assets, transparent cutouts, or when the user asks for no background.
            use_reference_images(boolean): Optional. Set true only to use images attached to this message or explicitly quoted/replied by it. False keeps text-to-image even when the prompt mentions editing. Earlier unquoted images are never used; if true has no selected images, ask the user to attach or quote them. Up to 16 images are used.
            output_format(string): Optional. png, jpeg, webp, or gif. Use gif for requested animated GIFs; this plugin currently supports text-only GIF generation, without reference images.
            frame_count(int): Optional. Integer GIF frame count, from 2 to 16; defaults to the configured value, initially 9.
            frame_duration_ms(int): Optional. Integer GIF milliseconds per frame, from 20 to 2000, rounded to 10 ms; initially 200. Smaller values play faster.

        """
        if not prompt or not prompt.strip():
            return "缺少图片提示词，无法生成或修改图片。"

        if not self._group_message_mentions_this_bot(event):
            logger.info("GPT Image tool call blocked in group because the source message did not mention this bot")
            return (
                "图片工具未执行：群聊中必须明确 @bot 才能触发生图或改图。"
                "没有生成图片，也没有开始处理图片。"
                "请直接用文字说明需要 @bot 后重新发送请求。"
            )

        raw_event_prompt = self._event_text(event) or prompt
        try:
            render_options = self._resolve_render_options(
                output_format=output_format, frame_count=frame_count,
                frame_duration_ms=frame_duration_ms, size=size,
                aspect_ratio=aspect_ratio, resolution=resolution,
                reference_image_paths=["requested-reference"] if use_reference_images else [],
            )
        except Exception as exc:
            return f"图片参数不可用：{self._friendly_error(exc)}"
        reference_paths: list[str] = []
        reference_source_records: list[dict[str, str]] = []
        if use_reference_images:
            reference_paths = await self._collect_reference_image_paths(
                event, stabilize=True, source_records=reference_source_records,
            )
            if not reference_paths:
                return "没有找到本条消息或明确引用消息中的参考图。请在本条消息附图，或回复/引用要修改的图片后再试；不会使用之前未引用的图片。"

        turn_key = self._tool_turn_key(event)
        turn_task = self._inflight_tool_turns.get(turn_key)
        if turn_task and not turn_task.done():
            trace = self._inflight_tool_turn_traces.get(turn_key)
            return self._tool_submitted_message(
                reference_count=len(reference_paths),
                flow_id=self._trace_flow_id(trace),
                include_plan_hint=True,
            )
        if turn_task and turn_task.done():
            self._inflight_tool_turns.pop(turn_key, None)
            self._inflight_tool_turn_traces.pop(turn_key, None)

        image_prompt = prompt
        dialog_prompt_passthrough = not reference_paths
        if dialog_prompt_passthrough and style and style.strip() and style.strip().lower() != "auto":
            image_prompt = f"{prompt}\n\nVisual style: {style.strip()}."

        request_key = self._tool_request_key(
            event,
            prompt=image_prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            quality=quality,
            style=style,
            transparent_background=transparent_background,
            reference_paths=reference_paths,
            render_options=render_options,
        )
        existing_task = self._inflight_tool_tasks.get(request_key)
        if existing_task and not existing_task.done():
            trace = self._inflight_tool_traces.get(request_key)
            return self._tool_submitted_message(
                reference_count=len(reference_paths),
                flow_id=self._trace_flow_id(trace),
                include_plan_hint=True,
            )
        if existing_task and existing_task.done():
            self._inflight_tool_tasks.pop(request_key, None)
            self._inflight_tool_traces.pop(request_key, None)

        wait_seconds = max(0, self._int_cfg("runtime", "llm_tool_foreground_wait_seconds", 0))
        trace = self._new_image_trace(
            event,
            trigger="llm_tool",
            operation="edit" if reference_paths else "generation",
            raw_user_prompt=raw_event_prompt,
            prompt=image_prompt,
            reference_image_paths=reference_paths,
            options={
                "size": size,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "quality": quality,
                "style": style,
                "transparent_background": transparent_background,
                "use_reference_images": use_reference_images,
                "dialog_prompt_passthrough": dialog_prompt_passthrough,
                "output_format": output_format,
                "frame_count": frame_count,
                "frame_duration_ms": frame_duration_ms,
                "render_options": render_options,
            },
        )
        route = "image_edit" if reference_paths else "text_to_image"
        reason = "llm_tool_reference_images" if reference_paths else "llm_tool_text_generation"
        if dialog_prompt_passthrough:
            reason = "llm_tool_dialog_prompt"
        self._trace_route_decision(
            trace,
            route=route,
            reason=reason,
            operation="edit" if reference_paths else "generation",
            reference_image_paths=reference_paths,
            llm_requested_reference_images=bool(use_reference_images),
        )
        self._trace_stage(
            trace, "reference_selection", status="succeeded",
            requested=bool(use_reference_images),
            policy="current_or_explicit_quote_only",
            sources=reference_source_records,
            selected_paths=reference_paths,
        )
        self._trace_stage(
            trace,
            "dialog_tool_prompt",
            status="succeeded",
            source="current_dialog_tool_call",
            input_prompt=raw_event_prompt,
            tool_prompt=prompt,
            output_prompt=image_prompt,
            style=style,
            passthrough=dialog_prompt_passthrough,
        )
        task = asyncio.create_task(
            self._generate_image(
                prompt=image_prompt,
                raw_user_prompt=raw_event_prompt,
                size=size,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                quality=quality,
                style=style,
                transparent_background=transparent_background,
                reference_image_paths=reference_paths,
                trace=trace,
                skip_prompt_enhance=dialog_prompt_passthrough,
                skip_prompt_translation=dialog_prompt_passthrough,
                raw_prompt_passthrough=dialog_prompt_passthrough,
                output_format=output_format,
                frame_count=frame_count,
                frame_duration_ms=frame_duration_ms,
                render_options=render_options,
            )
        )
        self._inflight_tool_tasks[request_key] = task
        self._inflight_tool_turns[turn_key] = task
        self._inflight_tool_traces[request_key] = trace
        self._inflight_tool_turn_traces[turn_key] = trace

        def _clear_inflight(done_task: asyncio.Task[tuple[str, str | None]]) -> None:
            if self._inflight_tool_tasks.get(request_key) is done_task:
                self._inflight_tool_tasks.pop(request_key, None)
                self._inflight_tool_traces.pop(request_key, None)
            if self._inflight_tool_turns.get(turn_key) is done_task:
                self._inflight_tool_turns.pop(turn_key, None)
                self._inflight_tool_turn_traces.pop(turn_key, None)

        task.add_done_callback(_clear_inflight)

        submitted_message = self._tool_submitted_message(
            reference_count=len(reference_paths),
            flow_id=self._trace_flow_id(trace),
            include_plan_hint=True,
        )

        try:
            if wait_seconds == 0:
                raise asyncio.TimeoutError
            image_path, revised_prompt = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=wait_seconds,
            )
        except asyncio.TimeoutError:
            error_prefix = "图片修改失败" if reference_paths else "图片生成失败"
            self._schedule_background_delivery(event, task, error_prefix=error_prefix, trace=trace)
            return submitted_message
        except Exception as exc:
            logger.error(
                "GPT Image tool failed (%s): %s",
                type(exc).__name__,
                self._friendly_error(exc),
            )
            error_prefix = "图片修改失败" if reference_paths else "图片生成失败"
            self._trace_fail(trace, exc)
            reply = await self._image_error_reply(event, exc, error_prefix)
            return f"流程ID：{self._trace_flow_id(trace)}\n{reply}"

        try:
            self._trace_stage(trace, "qq_delivery", status="started")
            await event.send(
                event.chain_result(
                    self._build_result_chain(
                        image_path,
                        revised_prompt,
                        flow_id=self._trace_flow_id(trace),
                        trace=trace,
                    )
                )
            )
            self._trace_stage(trace, "qq_delivery", status="succeeded")
        except Exception as exc:
            logger.error("GPT Image foreground delivery failed", exc_info=True)
            error_prefix = "图片修改失败" if reference_paths else "图片生成失败"
            self._trace_stage(trace, "qq_delivery", status="failed", error=self._friendly_error(exc))
            reply = await self._image_error_reply(event, exc, error_prefix)
            return f"流程ID：{self._trace_flow_id(trace)}\n{reply}"
        return None

    @filter.on_llm_request(priority=-9000)
    async def teach_llm_when_to_use_tool(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        guide = self._str_cfg("runtime", "tool_instruction", "") or DEFAULT_TOOL_GUIDE
        if guide.strip():
            req.system_prompt += f"\n\n{guide.strip()}\n"

    async def _generate_image(
        self,
        *,
        prompt: str,
        raw_user_prompt: str | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        transparent_background: bool = False,
        reference_image_paths: list[str] | None = None,
        trace: dict[str, Any] | None = None,
        skip_prompt_enhance: bool = False,
        skip_prompt_translation: bool = False,
        raw_prompt_passthrough: bool = False,
        advanced: bool = False,
        expected_image_selection: tuple[str, str, str] | None = None,
        output_format: str | None = None,
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
        render_options: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        # Freeze the model and connection together before any await. A provider
        # reload must never pair an old credential with a newly selected URL.
        self._trace_update(trace, status="running")
        connection = self._resolve_image_api_connection(advanced=advanced)
        model = connection["model"]
        base_url = connection["base_url"]
        render_options = dict(render_options) if render_options is not None else self._resolve_render_options(
            output_format=output_format, frame_count=frame_count,
            frame_duration_ms=frame_duration_ms, size=size,
            reference_image_paths=reference_image_paths,
            aspect_ratio=aspect_ratio, resolution=resolution,
        )
        if render_options["output_format"] == "gif" and not self._bool_cfg("gif", "enabled", False):
            raise RuntimeError("GIF 功能未开启。请在插件配置的“GIF 动图”中启用，并确认使用支持原生 GIF 的官方 API 源及完整透传参数的中转。")
        if render_options["output_format"] == "gif" and reference_image_paths:
            raise RuntimeError("本插件的 GIF 动图目前仅支持纯文字生成，不支持参考图或改图；本次不会丢弃参考图或改成静态图片。")
        if expected_image_selection is not None and (
            connection["provider_id"], model, base_url
        ) != expected_image_selection:
            raise RuntimeError("生图模型在确认后已变更，本次未提交；请重新建立 Plan 并确认。")

        references = self._prepare_reference_image_paths(reference_image_paths or [])
        self._trace_stage(
            trace,
            "reference_images",
            status="prepared",
            count=len(references),
            paths=references,
        )
        prompt_for_image = prompt
        style_for_prompt = style
        if not skip_prompt_enhance and self._should_enhance_prompt(bool(references)):
            self._trace_stage(
                trace,
                "prompt_enhance",
                status="started",
                provider_id=self._str_cfg("prompt_enhance", "provider_id", ""),
                mode=self._str_cfg("prompt_enhance", "mode", "placeholder"),
                input_prompt=prompt,
                raw_user_prompt=raw_user_prompt or prompt,
                reference_image_count=len(references),
            )
            try:
                enhanced_prompt = await self._enhance_prompt_with_reference_images(
                    prompt=prompt,
                    raw_user_prompt=raw_user_prompt or prompt,
                    style=style,
                    reference_image_paths=references,
                )
                if enhanced_prompt:
                    prompt_for_image = enhanced_prompt
                    # The enhancer sees the style value, so do not append the
                    # same style line twice after enhancement.
                    style_for_prompt = None
                    self._trace_stage(
                        trace,
                        "prompt_enhance",
                        status="succeeded",
                        output_prompt=enhanced_prompt,
                    )
            except Exception as exc:
                self._trace_stage(
                    trace,
                    "prompt_enhance",
                    status="failed",
                    error=self._friendly_error(exc),
                    fallback_to_original=self._bool_cfg("prompt_enhance", "fallback_to_original_prompt", True),
                )
                if self._bool_cfg("prompt_enhance", "fallback_to_original_prompt", True):
                    logger.warning(
                        "GPT Image prompt enhancement failed; fallback to original prompt: %s",
                        self._friendly_error(exc),
                        exc_info=True,
                    )
                else:
                    raise RuntimeError(f"提示词增强失败：{self._friendly_error(exc)}") from exc
        if raw_prompt_passthrough:
            final_prompt = prompt_for_image
        else:
            final_prompt = self._build_prompt(prompt_for_image, style_for_prompt)
        self._trace_stage(
            trace,
            "prompt_build",
            status="succeeded",
            input_prompt=prompt_for_image,
            style=style_for_prompt,
            output_prompt=final_prompt,
            passthrough=raw_prompt_passthrough,
        )
        if not skip_prompt_translation and self._bool_cfg("prompt", "translate_to_english", False):
            prompt_before_translation = final_prompt
            self._trace_stage(
                trace,
                "prompt_compression",
                status="started",
                provider_id=self._str_cfg("prompt", "translation_provider_id", ""),
                input_prompt=prompt_before_translation,
            )
            try:
                final_prompt = await self._translate_prompt_to_english(final_prompt)
                self._trace_stage(
                    trace,
                    "prompt_compression",
                    status="succeeded",
                    input_prompt=prompt_before_translation,
                    output_prompt=final_prompt,
                )
            except Exception as exc:
                self._trace_stage(
                    trace,
                    "prompt_compression",
                    status="failed",
                    input_prompt=prompt_before_translation,
                    error=self._friendly_error(exc),
                    fallback_to_original=self._bool_cfg("prompt", "translation_fallback_to_original", False),
                )
                if self._bool_cfg("prompt", "translation_fallback_to_original", False):
                    logger.warning(
                        "GPT Image prompt English translation failed; fallback to original prompt: %s",
                        self._friendly_error(exc),
                        exc_info=True,
                    )
                else:
                    raise RuntimeError(f"提示词英文化失败：{self._friendly_error(exc)}") from exc

        operation = "edit" if references else "generation"
        size, resolution, quality = self._apply_load_saving_options(
            operation=operation,
            raw_user_prompt=raw_user_prompt or prompt,
            size=size,
            resolution=resolution,
            quality=quality,
        )
        cache_key = (base_url, model, operation)
        self._trace_update(
            trace,
            operation=operation,
            final_prompt=final_prompt,
            api={
                "provider_id": connection["provider_id"],
                "base_url": base_url,
                "model": model,
                "generation_tier": "advanced" if advanced else "normal",
                "operation": operation,
            },
        )

        errors: list[ImageAPIError] = []
        response: dict[str, Any] | None = None
        succeeded_profile: str | None = None

        cached_profile = self._compat_profile_cache.get(cache_key)
        heuristic_profile = self._auto_detect_profile(
            operation=operation,
            size=size,
            aspect_ratio=aspect_ratio,
            reference_image_paths=references,
            base_url=base_url,
        )
        if self._request_has_strong_profile_hint(
            operation=operation,
            size=size,
            aspect_ratio=aspect_ratio,
            reference_image_paths=references,
            base_url=base_url,
        ):
            initial_profile = heuristic_profile
        else:
            initial_profile = cached_profile or heuristic_profile
        other_profile = "flexible" if initial_profile == "standard" else "standard"
        attempts = ["standard"] if render_options["output_format"] == "gif" else [initial_profile, other_profile]

        async with self._semaphore:
            cached_profile = self._compat_profile_cache.get(cache_key)
            for idx, profile in enumerate(attempts):
                payload = self._build_payload(
                    final_prompt,
                    model=model,
                    profile=profile,
                    size=size,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    quality=quality,
                    transparent_background=transparent_background,
                    reference_image_paths=references,
                    render_options=render_options,
                )
                self._log_request_start(
                    operation=operation,
                    profile=profile,
                    payload=payload,
                    base_url=base_url,
                )
                self._trace_stage(
                    trace,
                    "image_api",
                    status="started",
                    profile=profile,
                    operation=operation,
                    payload=self._trace_api_payload(payload),
                )
                try:
                    if references:
                        response = await self._post_images_edit_api(connection, payload, references)
                    else:
                        response = await self._post_images_generation_api(connection, payload)
                except ImageAPIError as exc:
                    errors.append(exc)
                    self._trace_stage(
                        trace,
                        "image_api",
                        status="failed",
                        profile=profile,
                        http_status=exc.status,
                        error=str(exc.message)[:600],
                        body=(exc.body or "")[:1200],
                        will_try_other_profile=idx + 1 < len(attempts) and exc.should_try_other_profile(),
                    )
                    logger.warning(
                        "GPT Image %s with profile=%s failed: HTTP %d %s",
                        operation,
                        profile,
                        exc.status,
                        str(exc.message)[:200],
                    )
                    if idx + 1 < len(attempts) and exc.should_try_other_profile():
                        # Worth a single retry with the other parameter dialect.
                        continue
                    break
                else:
                    succeeded_profile = profile
                    response_metadata = self._trace_response_metadata(response)
                    self._trace_stage(
                        trace,
                        "image_api",
                        status="succeeded",
                        profile=profile,
                        response_keys=sorted(response.keys()) if isinstance(response, dict) else [],
                        response_metadata=response_metadata,
                    )
                    break

        if response is None:
            if not errors:
                self._trace_update(trace, status="failed", error="图像接口请求失败：未知错误。")
                raise RuntimeError("图像接口请求失败：未知错误。")
            if len(errors) > 1:
                logger.error(
                    "GPT Image both profiles failed; first profile error HTTP %d: %s",
                    errors[0].status,
                    str(errors[0].message)[:300],
                )
            last = errors[-1]
            if render_options["output_format"] == "gif" and "gif output is not enabled" in f"{last.body} {last.message}".lower():
                error_message = f"图像接口 HTTP {last.status}：当前供应商尚未开放原生 GIF 输出，请选择支持 GIF 的供应商；本次不会改用静态图片。"
            else:
                error_message = self._format_api_error(last.status, last.body or last.message)
            self._trace_update(trace, status="failed", error=error_message)
            raise RuntimeError(error_message) from last

        if render_options["output_format"] != "gif" and succeeded_profile in {"standard", "flexible"}:
            self._compat_profile_cache[cache_key] = succeeded_profile
            if cached_profile and cached_profile != succeeded_profile:
                logger.info(
                    "GPT Image profile updated for %s | %s | %s: %s -> %s",
                    base_url,
                    model,
                    operation,
                    cached_profile,
                    succeeded_profile,
                )

        response_metadata = self._trace_response_metadata(response)
        image_path, revised_prompt = await self._save_image_from_response(
            response, expected_format=render_options["output_format"], expected_render=render_options,
        )
        self._cleanup_old_images()
        self._trace_update(
            trace,
            status="succeeded",
            result={
                "image_path": image_path,
                "revised_prompt": revised_prompt,
                "succeeded_profile": succeeded_profile,
                "response_headers": response_metadata.get("headers", {}),
            },
        )
        return image_path, revised_prompt

    async def _post_images_generation_api(
        self,
        connection: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = self._images_generation_url(connection["base_url"])
        headers = {**connection["headers"], "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._mutation_retry_times()

        for attempt in range(retry_times + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.post(url, headers=headers, json=payload, allow_redirects=False) as resp:
                        if 300 <= resp.status < 400:
                            raise RuntimeError("生图供应商返回了重定向，已停止发送凭据。请在 AstrBot 供应商设置中填写最终 API 地址。")
                        if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                            text = await self._read_limited_response_text(resp)
                            await self._sleep_before_network_retry(
                                attempt,
                                operation="GPT Image generation",
                                reason=f"HTTP {resp.status} {text[:200]}",
                                retry_total=retry_times,
                            )
                            continue
                        if resp.status >= 400:
                            text = await self._read_limited_response_text(resp)
                            raise self._build_api_error(resp.status, text)
                        text = await resp.text()
                        try:
                            payload = json.loads(text)
                            self._attach_http_response_metadata(payload, resp)
                            return payload
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("图像接口返回了无法解析的 JSON。") from exc
            except self._transient_network_exceptions() as exc:
                if attempt < retry_times:
                    await self._sleep_before_network_retry(
                        attempt,
                        operation="GPT Image generation",
                        reason=self._friendly_error(exc),
                        retry_total=retry_times,
                    )
                    continue
                raise RuntimeError(self._network_failure_message(exc, "图像接口请求失败")) from exc

        raise RuntimeError("图像接口请求失败：网络重试后仍未返回结果。")

    async def _post_images_edit_api(
        self,
        connection: dict[str, Any],
        payload: dict[str, Any],
        image_paths: list[str],
    ) -> dict[str, Any]:
        url = self._images_edit_url(connection["base_url"])
        headers = dict(connection["headers"])
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._mutation_retry_times()
        image_paths = self._prepare_reference_image_paths(image_paths)

        for attempt in range(retry_times + 1):
            form = aiohttp.FormData()
            for key, value in payload.items():
                form.add_field(key, self._multipart_field_value(value))

            opened_files = []
            try:
                for image_path in image_paths:
                    content_type = self._detect_image_content_type(image_path)
                    try:
                        file_obj = open(image_path, "rb")
                    except OSError as exc:
                        logger.warning(
                            "GPT Image skipped reference unavailable before upload "
                            "name=%s error=%s",
                            Path(image_path).name,
                            type(exc).__name__,
                        )
                        continue
                    opened_files.append(file_obj)
                    form.add_field(
                        "image",
                        file_obj,
                        filename=self._multipart_filename(image_path, content_type),
                        content_type=content_type,
                    )

                if not opened_files:
                    raise RuntimeError("参考图在上传前已失效，请重新发送或引用图片后再试。")

                try:
                    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                        async with session.post(url, headers=headers, data=form, allow_redirects=False) as resp:
                            if 300 <= resp.status < 400:
                                raise RuntimeError("生图供应商返回了重定向，已停止发送凭据。请在 AstrBot 供应商设置中填写最终 API 地址。")
                            if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                                text = await self._read_limited_response_text(resp)
                                await self._sleep_before_network_retry(
                                    attempt,
                                    operation="GPT Image edit",
                                    reason=f"HTTP {resp.status} {text[:200]}",
                                    retry_total=retry_times,
                                )
                                continue
                            if resp.status >= 400:
                                text = await self._read_limited_response_text(resp)
                                raise self._build_api_error(resp.status, text)
                            text = await resp.text()
                            try:
                                payload = json.loads(text)
                                self._attach_http_response_metadata(payload, resp)
                                return payload
                            except json.JSONDecodeError as exc:
                                raise RuntimeError("图像编辑接口返回了无法解析的 JSON。") from exc
                except self._transient_network_exceptions() as exc:
                    if attempt < retry_times:
                        await self._sleep_before_network_retry(
                            attempt,
                            operation="GPT Image edit",
                            reason=self._friendly_error(exc),
                            retry_total=retry_times,
                        )
                        continue
                    raise RuntimeError(self._network_failure_message(exc, "图像编辑接口请求失败")) from exc
            finally:
                for file_obj in opened_files:
                    try:
                        file_obj.close()
                    except Exception:
                        logger.debug("Failed to close GPT Image upload file", exc_info=True)

        raise RuntimeError("图像编辑接口请求失败：网络重试后仍未返回结果。")

    async def _save_image_from_response(
        self, response: dict[str, Any], *, expected_format: str | None = None,
        expected_render: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        normalized_response = self._extract_image_response(response)
        if normalized_response:
            response = normalized_response

        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("图像接口没有返回 data[0]。")

        item = data[0]
        if not isinstance(item, dict):
            raise RuntimeError("图像接口返回格式异常：data[0] 不是对象。")

        revised_prompt = item.get("revised_prompt")
        output_format = (expected_render or {}).get("output_format") or expected_format or self._output_format()
        suffix = ".jpg" if output_format == "jpeg" else f".{output_format}"
        file_path = self._image_dir / f"gpt_image_2_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"

        b64_json = item.get("b64_json")
        if isinstance(b64_json, str) and b64_json.strip():
            raw_b64 = b64_json.split(",", 1)[-1] if "," in b64_json[:64] else b64_json
            try:
                image_bytes = base64.b64decode(raw_b64)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError("图像接口返回的 b64_json 无法解码。") from exc
            if not image_bytes:
                raise RuntimeError("图像接口返回了空的 b64_json。")
            if output_format == "gif":
                self._validate_generated_gif(image_bytes, render_options=expected_render)
            actual_suffix = self._image_suffix_from_bytes(image_bytes)
            if actual_suffix:
                file_path = file_path.with_suffix(actual_suffix)
            file_path.write_bytes(image_bytes)
            return str(file_path), revised_prompt

        image_url = item.get("url")
        if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
            file_path = await self._download_image(
                image_url, file_path, expected_format=output_format, expected_render=expected_render,
            )
            return str(file_path), revised_prompt

        raise RuntimeError("图像接口没有返回 b64_json 或 url。")

    async def _download_image(
        self, url: str, file_path: Path, *, expected_format: str | None = None,
        expected_render: dict[str, Any] | None = None,
    ) -> Path:
        timeout = aiohttp.ClientTimeout(total=max(10, self._timeout_seconds()))
        retry_times = self._network_retry_times()
        headers = {"User-Agent": self._user_agent()}

        for attempt in range(retry_times + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status in TRANSIENT_HTTP_STATUSES and attempt < retry_times:
                            text = await self._read_limited_response_text(resp)
                            await self._sleep_before_network_retry(
                                attempt,
                                operation="GPT Image download",
                                reason=f"HTTP {resp.status} {text[:200]}",
                                retry_total=retry_times,
                            )
                            continue
                        if resp.status >= 400:
                            text = await self._read_limited_response_text(resp)
                            cf_message = self._cf_challenge_message(resp.status, text)
                            if cf_message:
                                raise RuntimeError(cf_message)
                            raise RuntimeError(f"下载图片失败：HTTP {resp.status} {text[:300]}")
                        data = await resp.read()
                        if not data:
                            raise RuntimeError("下载图片失败：响应内容为空。")
                        if expected_format == "gif":
                            self._validate_generated_gif(data, render_options=expected_render)
                        actual_suffix = self._image_suffix_from_bytes(data, resp.headers.get("Content-Type", ""))
                        if actual_suffix:
                            file_path = file_path.with_suffix(actual_suffix)
                        file_path.write_bytes(data)
                        return file_path
            except self._transient_network_exceptions() as exc:
                if attempt < retry_times:
                    await self._sleep_before_network_retry(
                        attempt,
                        operation="GPT Image download",
                        reason=self._friendly_error(exc),
                        retry_total=retry_times,
                    )
                    continue
                raise RuntimeError(self._network_failure_message(exc, "下载图片失败")) from exc

        raise RuntimeError("下载图片失败：网络重试后仍未返回结果。")

    async def _read_limited_response_text(self, resp: aiohttp.ClientResponse, limit: int = 2048) -> str:
        raw = await resp.content.read(limit + 1)
        truncated = len(raw) > limit
        if truncated:
            raw = raw[:limit]
        text = raw.decode(resp.charset or "utf-8", errors="replace")
        return f"{text}..." if truncated else text

    def _attach_http_response_metadata(self, payload: Any, resp: aiohttp.ClientResponse) -> None:
        if not isinstance(payload, dict):
            return
        headers: dict[str, str] = {}
        for key, value in resp.headers.items():
            normalized = key.lower()
            if normalized in HTTP_TRACE_HEADER_NAMES or normalized.endswith("request-id"):
                headers[key] = str(value)
        if headers:
            payload["_http_response_headers"] = headers

    def _multipart_field_value(self, value: Any) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)

    def _network_retry_times(self) -> int:
        return min(5, max(0, self._int_cfg("runtime", "retry_times", 1)))

    def _mutation_retry_times(self) -> int:
        configured = self._int_cfg(
            "runtime",
            "image_request_retry_times",
            0,
        )
        return min(5, max(0, configured))

    def _log_request_start(
        self, *, operation: str, profile: str, payload: dict[str, Any], base_url: str | None = None,
    ) -> None:
        url = self._images_edit_url(base_url) if operation == "edit" else self._images_generation_url(base_url)
        logger.info(
            "GPT Image request start operation=%s profile=%s model=%s url=%s timeout=%s retry=%s size=%s resolution=%s",
            operation,
            profile,
            payload.get("model", ""),
            url,
            self._timeout_seconds(),
            self._mutation_retry_times(),
            payload.get("size", ""),
            payload.get("resolution", ""),
        )

    def _transient_network_exceptions(self) -> tuple[type[BaseException], ...]:
        return (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionError)

    async def _sleep_before_network_retry(
        self,
        attempt: int,
        *,
        operation: str,
        reason: str,
        retry_total: int | None = None,
    ) -> None:
        delay = min(8.0, NETWORK_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        retry_total = self._network_retry_times() if retry_total is None else retry_total
        logger.warning(
            "%s transient network failure; retrying in %.1fs (%d/%d): %s",
            operation,
            delay,
            attempt + 1,
            retry_total,
            reason[:200],
        )
        await asyncio.sleep(delay)

    async def _image_error_reply(
        self,
        event: AstrMessageEvent,
        exc: Exception,
        error_prefix: str,
    ) -> str:
        if not self._is_safety_guardrail_error(exc):
            return f"{error_prefix}：{self._friendly_error(exc)}"

        fallback = self._safety_guardrail_fallback_reply(error_prefix)
        provider = self._event_dialog_provider(event)
        if not provider:
            return fallback

        operation = "图片修改" if "修改" in error_prefix else "图片生成"
        prompt = SAFETY_DIALOG_REPLY_PROMPT_TEMPLATE.format(operation=operation)
        try:
            timeout = max(5, self._int_cfg("runtime", "safety_reply_timeout_seconds", 20))
            response = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    session_id=f"gpt-image-safety-reply-{uuid.uuid4().hex}",
                    persist=False,
                ),
                timeout=timeout,
            )
            reply = self._clean_dialog_error_reply(getattr(response, "completion_text", "") or "")
            if reply:
                logger.info(
                    "GPT Image safety guardrail reply generated provider=%s",
                    self._provider_label(provider),
                )
                return reply
        except Exception as reply_exc:
            logger.warning("GPT Image safety guardrail dialog reply failed: %s", reply_exc)
        return fallback

    def _is_safety_guardrail_error(self, exc: Exception) -> bool:
        haystack = self._exception_chain_text(exc).lower()
        return any(keyword in haystack for keyword in SAFETY_GUARDRAIL_ERROR_KEYWORDS)

    def _exception_chain_text(self, exc: BaseException) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current and id(current) not in seen:
            seen.add(id(current))
            parts.append(f"{current.__class__.__name__}: {current}")
            for attr in ("message", "body"):
                value = getattr(current, attr, "")
                if value:
                    parts.append(str(value))
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        return "\n".join(parts)

    def _safety_guardrail_fallback_reply(self, error_prefix: str) -> str:
        operation = "图片修改" if "修改" in error_prefix else "图片生成"
        return (
            f"这次{operation}请求被上游安全规则拦截了，可能是提示词或参考图被判断为包含裸露、"
            "性暗示或成人向内容。可以把需求改成更明确的非成人、非裸露描述，比如补充服装、"
            "材质、构图、风格和动作后再试。"
        )

    def _clean_dialog_error_reply(self, text: str) -> str:
        cleaned = self._clean_enhanced_prompt(text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:500].rstrip()

    def _schedule_background_delivery(
        self,
        event: AstrMessageEvent,
        task: asyncio.Task[tuple[str, str | None]],
        *,
        error_prefix: str = "图片生成失败",
        trace: dict[str, Any] | None = None,
    ) -> None:
        # Keep a strong reference to the background sender. Long image jobs can
        # outlive the tool call, and losing this task would mean the image is
        # generated but never sent back to the chat.
        async def _deliver():
            try:
                image_path, revised_prompt = await task
                self._trace_stage(trace, "qq_delivery", status="started")
                await event.send(
                    event.chain_result(
                        self._build_result_chain(
                            image_path,
                            revised_prompt,
                            flow_id=self._trace_flow_id(trace),
                            trace=trace,
                        )
                    )
                )
                self._trace_stage(trace, "qq_delivery", status="succeeded")
            except Exception as exc:
                self._trace_fail(trace, exc)
                logger.error(
                    "GPT Image background delivery failed (%s): %s",
                    type(exc).__name__,
                    self._friendly_error(exc),
                )
                try:
                    reply = await self._image_error_reply(event, exc, error_prefix)
                    flow_id = self._trace_flow_id(trace)
                    if flow_id:
                        reply = f"流程ID：{flow_id}\n{reply}"
                    await event.send(event.plain_result(reply))
                except Exception as send_exc:
                    logger.error(
                        "Failed to send GPT Image background error (%s): %s",
                        type(send_exc).__name__,
                        self._friendly_error(send_exc),
                    )

        sender = asyncio.create_task(_deliver())
        self._background_tasks.add(sender)

        def _report_done(done_task: asyncio.Task):
            self._background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc:
                logger.error(
                    "GPT Image background sender crashed (%s): %s",
                    type(exc).__name__,
                    self._friendly_error(exc),
                )

        sender.add_done_callback(_report_done)

    def _build_payload(
        self,
        prompt: str,
        *,
        profile: str,
        size: str | None,
        aspect_ratio: str | None,
        resolution: str | None,
        quality: str | None,
        transparent_background: bool,
        reference_image_paths: list[str],
        model: str | None = None,
        render_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = model or self._model_name()
        render_options = render_options or self._resolve_render_options(size=size)
        output_format = render_options["output_format"]
        # Recheck after queueing / prompt awaits: disabling GIF must also stop
        # previously prepared jobs before their first paid request.
        if output_format == "gif" and not self._bool_cfg("gif", "enabled", False):
            raise RuntimeError("GIF 功能未开启。请在插件配置的“GIF 动图”中启用，并确认使用支持原生 GIF 的官方 API 源及完整透传参数的中转。")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
        }

        # ``standard`` keeps the OpenAI-strict size grammar (pixel sizes only,
        # ratios mapped to one of three OpenAI-accepted dimensions). ``flexible``
        # lets the size field carry an aspect ratio string the way most webpage
        # reverse / 2api suppliers expect.
        if output_format == "gif":
            normalized_size = render_options["size"]
        elif profile == "flexible":
            normalized_size = self._normalize_compatible_size(
                size,
                aspect_ratio=aspect_ratio,
                reference_image_paths=reference_image_paths,
                prefer_reference_ratio=bool(reference_image_paths),
            )
        else:
            normalized_size = self._normalize_openai_size(
                size,
                aspect_ratio=aspect_ratio,
                reference_image_paths=reference_image_paths,
                prefer_reference_ratio=bool(reference_image_paths),
            )
        if normalized_size != "auto":
            payload["size"] = normalized_size

        normalized_ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=size,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=bool(reference_image_paths),
        )
        normalized_resolution = self._normalize_resolution(resolution)
        if output_format != "gif" and normalized_resolution != "auto":
            payload["resolution"] = normalized_resolution

        normalized_quality = self._normalize_quality(quality)
        if normalized_quality != "auto":
            payload["quality"] = normalized_quality

        configured_background = self._str_cfg("image", "background", "auto")
        if output_format == "jpeg" and (transparent_background or configured_background == "transparent"):
            output_format = "png"
        payload["output_format"] = output_format
        if output_format == "gif":
            payload["frame_count"] = render_options["frame_count"]
            payload["frame_duration_ms"] = render_options["frame_duration_ms"]

        compression = self._int_cfg("image", "output_compression", 0)
        if output_format in {"jpeg", "webp"} and 0 < compression <= 100:
            payload["output_compression"] = compression

        background = "transparent" if transparent_background else configured_background
        background = background if background in VALID_BACKGROUNDS else "auto"
        if background != "auto":
            payload["background"] = background

        moderation = self._str_cfg("image", "moderation", "")
        if moderation:
            payload["moderation"] = moderation

        return payload

    def _should_enhance_prompt(self, has_reference_images: bool) -> bool:
        if not self._bool_cfg("prompt_enhance", "enable", False):
            return False
        apply_to = self._str_cfg("prompt_enhance", "apply_to", "edit_only").lower()
        if apply_to == "edit_only" and not has_reference_images:
            return False
        return True

    def _event_dialog_provider(self, event: AstrMessageEvent) -> Provider | None:
        provider = None
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        except Exception:
            provider = None
        if not provider:
            try:
                provider = self.context.get_using_provider()
            except Exception:
                provider = None
        return provider if isinstance(provider, Provider) else None

    def _collect_reference_description_texts(self, event: AstrMessageEvent) -> list[str]:
        texts: list[str] = []

        def add_text(value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in texts:
                texts.append(text)

        def walk_reply(component: Any) -> None:
            if isinstance(component, Reply):
                add_text(getattr(component, "message_str", ""))
                add_text(getattr(component, "text", ""))
                walk_chain(getattr(component, "chain", None))

        def walk_chain(chain: Any) -> None:
            if isinstance(chain, Plain):
                add_text(getattr(chain, "text", ""))
                return
            if isinstance(chain, Reply):
                walk_reply(chain)
                return
            if not isinstance(chain, (list, tuple)):
                return

            for component in chain:
                if isinstance(component, Reply):
                    walk_reply(component)
                    continue
                nested_chain = getattr(component, "chain", None)
                if nested_chain:
                    walk_chain(nested_chain)
                nested_content = getattr(component, "content", None)
                if nested_content:
                    walk_chain(nested_content)
                nested_nodes = getattr(component, "nodes", None)
                if nested_nodes:
                    for node in nested_nodes:
                        walk_chain(getattr(node, "content", None))

        try:
            message_chain = event.get_messages()
        except Exception:
            logger.debug("Failed to read AstrBot message chain for quoted text", exc_info=True)
            message_chain = getattr(getattr(event, "message_obj", None), "message", [])
        walk_chain(message_chain)
        return texts

    async def _enhance_prompt_from_description_reference(
        self,
        *,
        prompt: str,
        reference_image_paths: list[str],
        reference_texts: list[str],
    ) -> str:
        provider = self._prompt_enhance_provider()
        if not provider:
            raise RuntimeError("没有可用的提示词增强模型提供商")

        max_images = max(0, self._int_cfg("prompt_enhance", "max_reference_images", 1))
        image_urls = reference_image_paths[:max_images] if max_images else []
        reference_text = "\n\n".join(text.strip() for text in reference_texts if text.strip())
        enhancer_prompt = f"""
你是图像生成提示词工程师。用户引用/发送了一张可能包含文字描述的图片，或引用了一段文字，并要求“按这个描述生成一张图”。

用户原始需求：
{prompt}

引用消息文字（如果有）：
{reference_text or "无"}

参考图片数量：
{len(image_urls)}

请遵守：
1. 如果参考图片里有文字，先准确识别图片中的文字内容，并把它作为主要图像描述来源。
2. 如果引用消息文字不为空，也把引用文字作为主要图像描述来源。
3. 这是纯文字生图任务，不是改图任务；不要要求保留参考图片的构图、背景、人物或画风，除非图片内文字本身这么要求。
4. 把识别到的描述整理成一段完整、具体、可直接提交给图像生成模型的最终提示词。
5. 不要新增无关文字、logo、水印或额外角色。
6. 只输出最终图像提示词，不要解释，不要 Markdown，不要代码块。
""".strip()
        timeout = max(5, self._int_cfg("prompt_enhance", "timeout_seconds", 60))
        logger.info(
            "GPT Image description reference prompt enhancement start provider=%s images=%d quoted_text_len=%d timeout=%s",
            self._provider_label(provider),
            len(image_urls),
            len(reference_text),
            timeout,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=enhancer_prompt,
                session_id=f"gpt-image-description-reference-{uuid.uuid4().hex}",
                image_urls=image_urls,
                persist=False,
            ),
            timeout=timeout,
        )
        enhanced = self._clean_enhanced_prompt(getattr(response, "completion_text", "") or "")
        if not enhanced:
            raise RuntimeError("提示词增强模型没有返回有效内容")

        max_chars = max(200, self._int_cfg("prompt_enhance", "max_output_chars", 4000))
        enhanced = enhanced[:max_chars].rstrip()
        logger.info(
            "GPT Image description reference prompt enhancement done provider=%s images=%d output_len=%d",
            self._provider_label(provider),
            len(image_urls),
            len(enhanced),
        )
        return enhanced

    async def _enhance_prompt_with_reference_images(
        self,
        *,
        prompt: str,
        raw_user_prompt: str,
        style: str | None,
        reference_image_paths: list[str],
    ) -> str:
        provider = self._prompt_enhance_provider()
        if not provider:
            raise RuntimeError("没有可用的提示词增强模型提供商")

        max_images = max(0, self._int_cfg("prompt_enhance", "max_reference_images", 1))
        image_urls = reference_image_paths[:max_images] if max_images else []
        mode = self._str_cfg("prompt_enhance", "mode", "placeholder").lower()
        use_placeholder_mode = mode in {"placeholder", "fill_placeholder", "template_fill"}
        template = (
            self._str_cfg(
                "prompt_enhance",
                "prompt_template",
                "",
            )
            or DEFAULT_PROMPT_ENHANCE_TEMPLATE
        )
        enhancer_prompt = self._render_prompt_enhance_template(
            template,
            raw_user_prompt=raw_user_prompt or prompt,
            user_prompt=prompt,
            style=style or "",
            image_count=str(len(image_urls)),
        )
        placeholder = self._prompt_enhance_placeholder()
        placeholder_fill = self._placeholder_fill_text(raw_user_prompt, prompt)
        if use_placeholder_mode:
            enhancer_prompt = self._with_placeholder_fill_instruction(
                enhancer_prompt,
                placeholder=placeholder,
                placeholder_fill=placeholder_fill,
                raw_user_prompt=raw_user_prompt or prompt,
                user_prompt=prompt,
            )
        timeout = max(5, self._int_cfg("prompt_enhance", "timeout_seconds", 60))

        logger.info(
            "GPT Image prompt enhancement start mode=%s provider=%s images=%d raw_len=%d prompt_len=%d timeout=%s",
            "placeholder" if use_placeholder_mode else "full_prompt",
            self._provider_label(provider),
            len(image_urls),
            len(raw_user_prompt or ""),
            len(prompt or ""),
            timeout,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=enhancer_prompt,
                session_id=f"gpt-image-prompt-enhance-{uuid.uuid4().hex}",
                image_urls=image_urls,
                persist=False,
            ),
            timeout=timeout,
        )

        enhanced_text = self._clean_enhanced_prompt(getattr(response, "completion_text", "") or "")
        if not enhanced_text:
            raise RuntimeError("提示词增强模型返回了空内容")

        max_chars = max(200, self._int_cfg("prompt_enhance", "max_output_chars", 4000))
        if len(enhanced_text) > max_chars:
            enhanced_text = enhanced_text[:max_chars].rstrip()

        enhanced = enhanced_text
        if use_placeholder_mode:
            enhanced = self._replace_prompt_placeholder(
                enhanced_text,
                placeholder=placeholder,
                placeholder_fill=placeholder_fill,
            )
        placeholder_left = placeholder in enhanced

        logger.info(
            "GPT Image prompt enhancement done mode=%s provider=%s images=%d output_len=%d placeholder_left=%s",
            "placeholder" if use_placeholder_mode else "full_prompt",
            self._provider_label(provider),
            len(image_urls),
            len(enhanced),
            placeholder_left,
        )
        return enhanced

    def _prompt_enhance_provider(self) -> Provider | None:
        provider_id = self._str_cfg("prompt_enhance", "provider_id", "")
        if not provider_id and self._bool_cfg("prompt_enhance", "use_global_image_caption_provider", True):
            provider_id = self._global_image_caption_provider_id()
            if provider_id:
                logger.info("GPT Image prompt enhance using global image caption provider: %s", provider_id)
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise RuntimeError(f"没有找到 ID 为 {provider_id} 的提示词增强模型提供商")
        else:
            provider = self.context.get_using_provider()
            if not provider:
                raise RuntimeError("当前默认模型提供商不可用，且未配置提示词增强模型 provider_id")

        if not isinstance(provider, Provider):
            raise RuntimeError(f"提示词增强模型提供商类型错误：{type(provider)}")
        return provider

    def _global_image_caption_provider_id(self) -> str:
        try:
            config = self.context.get_config()
            provider_settings = config.get("provider_settings", {}) if isinstance(config, dict) else {}
            return str(provider_settings.get("default_image_caption_provider_id") or "").strip()
        except Exception as exc:
            logger.debug("GPT Image failed to read global image caption provider: %s", exc)
        return ""

    def _prompt_translation_provider(self) -> Provider | None:
        provider_id = self._str_cfg("prompt", "translation_provider_id", "")
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise RuntimeError(f"没有找到 ID 为 {provider_id} 的提示词英文化模型提供商")
        else:
            provider = self.context.get_using_provider()
            if not provider:
                raise RuntimeError("当前默认模型提供商不可用，且未配置提示词英文化模型 provider_id")

        if not isinstance(provider, Provider):
            raise RuntimeError(f"提示词英文化模型提供商类型错误：{type(provider)}")
        return provider

    async def _translate_prompt_to_english(
        self,
        prompt: str,
        *,
        max_output_chars: int | None = None,
    ) -> str:
        provider = self._prompt_translation_provider()
        template = (
            self._str_cfg("prompt", "translation_template", "")
            or DEFAULT_PROMPT_TRANSLATION_TEMPLATE
        )
        translation_prompt = template.replace("{prompt}", prompt or "").strip()
        if "{prompt}" not in template:
            translation_prompt = f"{translation_prompt}\n\nFinal image prompt:\n{prompt or ''}".strip()

        timeout = max(5, self._int_cfg("prompt", "translation_timeout_seconds", 45))
        logger.info(
            "GPT Image prompt English translation start provider=%s prompt_len=%d timeout=%s",
            self._provider_label(provider),
            len(prompt or ""),
            timeout,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=translation_prompt,
                session_id=f"gpt-image-prompt-english-{uuid.uuid4().hex}",
                persist=False,
            ),
            timeout=timeout,
        )
        translated = self._clean_enhanced_prompt(getattr(response, "completion_text", "") or "")
        if not translated:
            raise RuntimeError("提示词英文化模型没有返回有效内容")
        max_chars = max(
            200,
            max_output_chars
            if max_output_chars is not None
            else self._int_cfg("prompt", "translation_max_output_chars", 4000),
        )
        translated = translated[:max_chars].rstrip()
        logger.info(
            "GPT Image prompt English translation done provider=%s output_len=%d",
            self._provider_label(provider),
            len(translated),
        )
        return translated

    def _render_prompt_enhance_template(self, template: str, **values: str) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value or "")
        if "{user_prompt}" not in template and "{raw_user_prompt}" not in template:
            rendered = (
                f"{rendered.strip()}\n\n用户原始消息：\n{values.get('raw_user_prompt', '')}"
                f"\n\n工具改写提示词：\n{values.get('user_prompt', '')}"
            )
        return rendered.strip()

    def _prompt_enhance_placeholder(self) -> str:
        return (
            self._str_cfg("prompt_enhance", "placeholder", DEFAULT_PROMPT_SUBJECT_PLACEHOLDER)
            or DEFAULT_PROMPT_SUBJECT_PLACEHOLDER
        )

    def _placeholder_fill_text(self, raw_user_prompt: str, prompt: str) -> str:
        fill_text = (raw_user_prompt or prompt or "").strip()
        fill_text = re.sub(r"^\s*/(?:生图|画图|绘图|生成|改图|修图|gpt\s*image)\s*", "", fill_text, flags=re.I)
        fill_text = re.sub(r"^\s*@\S+\s*", "", fill_text)
        fill_text = fill_text.strip()
        if not fill_text:
            fill_text = (prompt or "").strip()
        return fill_text[:1200].strip()

    def _with_placeholder_fill_instruction(
        self,
        rendered_template: str,
        *,
        placeholder: str,
        placeholder_fill: str,
        raw_user_prompt: str,
        user_prompt: str,
    ) -> str:
        return f"""
{rendered_template.strip()}

---
本次插件运行模式：占位符填充。

上面的“完整提示词增强模板”是用户配置的主模板。请严格按它分析参考图的风格、构图、质感和需要保留的元素；但如果模板要求在输出里使用占位符 `{placeholder}`，这一次不要保留占位符，而是把它替换成用户本次真正想生成或修改的主体内容。

用户本次原始要求：
{raw_user_prompt}

工具整理后的要求：
{user_prompt}

请把 `{placeholder}` 替换为下面这段需求所表达的主体/修改目标，必要时用更适合绘图模型的中文表述重写，但不要改变用户意图：
{placeholder_fill}

要求：
1. 最终只输出一段将直接提交给 GPT Image 的完整提示词。
2. 输出中不得再出现 `{placeholder}`。
3. 参考图用于理解画风、构图、背景、文字和应保留的细节；如果用户要求替换人物、主体、服装、动作或场景，以用户本次要求为准，不要把参考图原始主体当成新主体。
4. 不要解释，不要 Markdown，不要代码块。
""".strip()

    def _replace_prompt_placeholder(
        self,
        prompt: str,
        *,
        placeholder: str,
        placeholder_fill: str,
    ) -> str:
        if placeholder and placeholder in prompt:
            return prompt.replace(placeholder, placeholder_fill).strip()
        return prompt.strip()

    def _clean_enhanced_prompt(self, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return cleaned.strip().strip('"').strip("'").strip()

    def _provider_label(self, provider: Provider) -> str:
        for attr in ("id", "provider_id", "name", "model"):
            value = getattr(provider, attr, None)
            if value:
                return str(value)
        return provider.__class__.__name__

    def _build_prompt(self, prompt: str, style: str | None) -> str:
        parts = []
        prefix = self._str_cfg("prompt", "prompt_prefix", "")
        suffix = self._str_cfg("prompt", "prompt_suffix", "")
        negative = self._str_cfg("prompt", "negative_prompt", "")

        if prefix:
            parts.append(prefix.strip())
        parts.append(prompt.strip())
        if style and style.strip() and style.strip().lower() != "auto":
            parts.append(f"Visual style: {style.strip()}.")
        if suffix:
            parts.append(suffix.strip())
        if negative:
            parts.append(f"Avoid: {negative.strip()}.")

        return "\n\n".join(parts).strip()

    def _build_result_chain(
        self,
        image_path: str,
        revised_prompt: str | None,
        *,
        flow_id: str | None = None,
        trace: dict[str, Any] | None = None,
    ) -> list[Any]:
        del revised_prompt
        detail_text = self._build_result_detail_text(
            image_path=image_path,
            flow_id=flow_id,
            trace=trace,
        )
        return [
            Image.fromFileSystem(image_path),
            Plain(f"\n{detail_text}"),
        ]

    def _build_result_detail_text(
        self,
        *,
        image_path: str,
        flow_id: str | None,
        trace: dict[str, Any] | None,
    ) -> str:
        lines: list[str] = []
        if flow_id:
            lines.append(f"流程ID：{flow_id}")
        lines.append(f"分辨率：{self._output_resolution_text(image_path)}")
        elapsed_text = self._trace_elapsed_text(trace)
        if elapsed_text:
            lines.append(f"耗时：{elapsed_text}")
        return "\n".join(lines)

    def _output_resolution_text(self, image_path: str) -> str:
        dimensions = self._image_dimensions(image_path)
        if not dimensions:
            return "未知"
        width, height = dimensions
        if width <= 0 or height <= 0:
            return "未知"
        return f"{width}x{height}"

    def _trace_elapsed_text(self, trace: dict[str, Any] | None) -> str:
        if not isinstance(trace, dict):
            return ""
        try:
            created_at = float(trace.get("created_at") or 0)
        except (TypeError, ValueError):
            created_at = 0
        if created_at <= 0:
            return ""
        return self._format_elapsed_seconds(time.time() - created_at)

    def _format_elapsed_seconds(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}小时{minutes}分{secs}秒"
        if minutes:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"

    def _tool_submitted_message(
        self,
        *,
        reference_count: int = 0,
        flow_id: str | None = None,
        include_plan_hint: bool = False,
    ) -> str:
        del reference_count
        message = "收到，开始画了，别催，如果等会提示报错喊老大排查"
        if flow_id:
            message += f"\n流程ID：{flow_id}"
        if include_plan_hint:
            message += "\n复杂需求可以用 /plan 进入高级画图模式。"
        return message

    def _plan_session_key(self, event: AstrMessageEvent) -> str:
        group_id = self._event_identity_value(event, ("group_id",), ("get_group_id",))
        sender_id = self._event_identity_value(event, ("sender_id", "user_id"), ("get_sender_id",))
        session_id = self._event_identity_value(event, ("session_id",), ("get_session_id",))

        if group_id:
            return f"group:{group_id}:user:{sender_id or session_id or 'unknown'}"
        return f"private:{sender_id or session_id or self._event_scope_key(event)}"

    def _plan_owner_info(self, event: AstrMessageEvent) -> dict[str, str]:
        group_id = self._event_identity_value(event, ("group_id",), ("get_group_id",))
        sender_id = self._event_identity_value(event, ("sender_id", "user_id"), ("get_sender_id",))
        session_id = self._event_identity_value(event, ("session_id",), ("get_session_id",))
        owner_id = sender_id or session_id or "unknown"
        if group_id:
            label = f"群 {group_id} / QQ {owner_id}"
            scope = "group"
        else:
            label = f"私聊 / QQ {owner_id}" if owner_id != "unknown" else "私聊 / 未知用户"
            scope = "private"
        return {
            "scope": scope,
            "group_id": group_id,
            "sender_id": sender_id,
            "session_id": session_id,
            "owner_id": owner_id,
            "label": label,
        }

    def _plan_owner_text(self, session: dict[str, Any] | None, event: AstrMessageEvent | None = None) -> str:
        owner = session.get("owner") if isinstance(session, dict) else None
        if isinstance(owner, dict):
            label = str(owner.get("label") or "").strip()
            if label:
                return label
        if event is not None:
            return self._plan_owner_info(event).get("label", "未知用户")
        return "未知用户"

    def _mark_plan_event(self, event: AstrMessageEvent) -> bool:
        now = time.time()
        for key, handled_at in list(self._plan_handled_events.items()):
            if now - handled_at > 30:
                self._plan_handled_events.pop(key, None)

        marker = self._tool_turn_key(event)
        handled_at = self._plan_handled_events.get(marker)
        if handled_at and now - handled_at < 5:
            return False
        self._plan_handled_events[marker] = now
        return True

    def _plan_session_ttl_seconds(self) -> int:
        return max(0, self._int_cfg("plan", "session_ttl_seconds", 15 * 60))

    def _cleanup_expired_plan_sessions(self, *, current_key: str | None = None) -> bool:
        ttl = self._plan_session_ttl_seconds()
        if ttl <= 0 or not current_key:
            return False

        now = time.time()
        session = self._plan_sessions.get(current_key)
        if not session:
            return False
        try:
            updated_at = float(session.get("updated_at") or session.get("created_at") or 0)
        except Exception:
            updated_at = 0
        if updated_at <= 0 or now - updated_at > ttl:
            self._plan_sessions.pop(current_key, None)
            return True
        return False

    def _plan_expired_text(self) -> str:
        minutes = max(1, self._plan_session_ttl_seconds() // 60)
        return f"当前 GPT Image · 龙绘 Plan 已超过 {minutes} 分钟未更新，已自动退出。请重新用 /plan 开始。"

    def _plan_help_text(self) -> str:
        return (
            "GPT Image · 龙绘 Plan 模式：\n"
            "/plan 画图需求 - 创建或补充高级画图计划\n"
            "/plan status - 查看当前计划\n"
            "/plan confirm - 确认并开始生成\n"
            "/plan quit - 退出当前计划\n"
            "可以在 /plan 消息里附带或引用参考图。"
        )

    def _plan_raw_prompt(self, session: dict[str, Any] | None) -> str:
        if not isinstance(session, dict):
            return ""
        prompts = session.get("prompts") or []
        if not isinstance(prompts, list):
            return str(prompts or "").strip()
        return "\n".join(str(item).strip() for item in prompts if str(item).strip()).strip()

    def _merge_plan_options(self, session: dict[str, Any], opts: dict[str, Any]) -> None:
        target = session.setdefault("options", {})
        if opts.get("output_format") and str(opts["output_format"]).lower() != "gif":
            target.pop("frame_count", None)
            target.pop("frame_duration_ms", None)
        for key, value in (opts or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, bool) and value is False:
                continue
            target[key] = value

    def _freeze_plan_render_options(self, session: dict[str, Any]) -> dict[str, Any]:
        """Keep the plan's displayed rendering defaults until explicitly edited.

        Args:
            session: The active plan whose options are frozen in place.

        Returns:
            An independent rendering snapshot without provider credentials.
        """
        options = session.setdefault("options", {})
        render = self._resolve_render_options(
            output_format=options.get("output_format"),
            frame_count=options.get("frame_count"),
            frame_duration_ms=options.get("frame_duration_ms"),
            size=options.get("size"),
            reference_image_paths=session.get("reference_image_paths"),
            aspect_ratio=options.get("aspect_ratio"), resolution=options.get("resolution"),
        )
        options.update(render)
        return render

    def _merge_plan_references(self, session: dict[str, Any], reference_paths: list[str]) -> None:
        if not reference_paths:
            return
        target = session.setdefault("reference_image_paths", [])
        seen = {os.path.abspath(str(path)) for path in target if path}
        for path in reference_paths:
            normalized = os.path.abspath(str(path))
            if normalized in seen:
                continue
            seen.add(normalized)
            target.append(normalized)
            if len(target) >= self._max_reference_images():
                break

    async def _refresh_plan_session(self, session: dict[str, Any], *, latest_prompt: str) -> None:
        revision = session.get("revision")
        if not self._bool_cfg("plan", "enable_model_planning", True):
            session["planned_prompt"] = self._plan_raw_prompt(session)
            session["planned_response"] = self._fallback_plan_response(session)
            session.pop("planning_error", None)
            return

        provider = self._plan_provider()
        raw_prompt = self._plan_raw_prompt(session)
        opts = session.get("options") or {}
        refs = [
            str(path)
            for path in session.get("reference_image_paths", [])
            if path and os.path.exists(str(path))
        ][: self._max_reference_images()]
        max_plan_images = max(0, self._int_cfg("plan", "max_reference_images", 4))
        image_urls = refs[:max_plan_images] if max_plan_images else []
        previous_plan = str(session.get("planned_response") or "").strip()
        prompt = self._render_plan_prompt(
            raw_prompt=raw_prompt,
            latest_prompt=latest_prompt,
            previous_plan=previous_plan,
            options=self._format_plan_options(opts) or "默认",
            reference_count=len(refs),
        )
        timeout = max(5, self._int_cfg("plan", "timeout_seconds", 60))
        logger.info(
            "GPT Image · 龙绘 plan model start provider=%s refs=%d prompt_len=%d timeout=%s",
            self._provider_label(provider),
            len(image_urls),
            len(raw_prompt),
            timeout,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                session_id=f"image2-plan-{uuid.uuid4().hex}",
                image_urls=image_urls,
                persist=False,
            ),
            timeout=timeout,
        )
        planned_response = self._clean_enhanced_prompt(getattr(response, "completion_text", "") or "")
        if session.get("revision") != revision:
            raise RuntimeError("计划已修改，丢弃旧整理结果")
        if not planned_response:
            raise RuntimeError("Plan 模型没有返回有效内容")
        max_chars = max(500, self._int_cfg("plan", "max_output_chars", 5000))
        planned_response = planned_response[:max_chars].rstrip()
        planned_prompt = self._extract_plan_final_prompt(planned_response) or raw_prompt
        session["planned_response"] = planned_response
        session["planned_prompt"] = planned_prompt
        session["planned_at"] = time.time()
        session.pop("planning_error", None)
        logger.info(
            "GPT Image · 龙绘 plan model done provider=%s output_len=%d final_prompt_len=%d",
            self._provider_label(provider),
            len(planned_response),
            len(planned_prompt),
        )

    def _plan_provider(self) -> Provider:
        provider_id = self._str_cfg("plan", "provider_id", "")
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise RuntimeError(f"没有找到 ID 为 {provider_id} 的 Plan 模型提供商")
            if not isinstance(provider, Provider):
                raise RuntimeError(f"Plan 模型提供商类型错误：{type(provider)}")
            return provider

        provider = self.context.get_using_provider()
        if not isinstance(provider, Provider):
            raise RuntimeError("当前默认模型提供商不可用，且未配置 Plan 模型 provider_id")
        return provider

    def _render_plan_prompt(
        self,
        *,
        raw_prompt: str,
        latest_prompt: str,
        previous_plan: str,
        options: str,
        reference_count: int,
    ) -> str:
        template = self._str_cfg("plan", "prompt_template", "") or DEFAULT_PLAN_PROMPT_TEMPLATE
        values = {
            "raw_prompt": raw_prompt or "（无文字需求）",
            "latest_prompt": latest_prompt or "（本次未追加文字，仅可能追加了参考图或参数）",
            "previous_plan": previous_plan or "（暂无）",
            "options": options or "默认",
            "reference_count": str(reference_count),
        }
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        if "{raw_prompt}" not in template:
            rendered = f"{rendered.strip()}\n\n用户累计需求：\n{values['raw_prompt']}"
        return rendered.strip()

    def _extract_plan_final_prompt(self, planned_response: str) -> str:
        text = self._clean_enhanced_prompt(planned_response)
        if not text:
            return ""

        tag_match = re.search(r"<final_prompt>\s*(.*?)\s*</final_prompt>", text, flags=re.I | re.S)
        if tag_match:
            return self._clean_enhanced_prompt(tag_match.group(1))

        marker_patterns = [
            r"【最终提示词】\s*(.*?)(?=\n\s*【[^】]+】|\Z)",
            r"最终提示词\s*[:：]\s*(.*?)(?=\n\s*(?:【[^】]+】|[^\n]{0,20}[:：])|\Z)",
            r"Final Prompt\s*[:：]\s*(.*?)(?=\n\s*(?:[A-Z][A-Za-z ]{0,30}[:：]|【[^】]+】)|\Z)",
        ]
        for pattern in marker_patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                return self._clean_enhanced_prompt(match.group(1))
        return ""

    def _plan_final_prompt(self, session: dict[str, Any] | None) -> str:
        if not isinstance(session, dict):
            return ""
        planned_prompt = str(session.get("planned_prompt") or "").strip()
        if planned_prompt:
            return planned_prompt
        return self._extract_plan_final_prompt(str(session.get("planned_response") or ""))

    def _plan_submission_fingerprint(self, session: dict[str, Any]) -> str:
        connection = self._resolve_image_api_connection(advanced=True)
        inputs = {
            "render_options": self._freeze_plan_render_options(session),
            "image_selection": (connection["provider_id"], connection["model"], connection["base_url"]),
            "raw_prompt": self._plan_raw_prompt(session),
            "planned_prompt": self._plan_final_prompt(session),
            "options": session.get("options") or {},
            "reference_image_paths": session.get("reference_image_paths") or [],
            "revision": session.get("revision", 0),
            "prompt_settings": {
                key: self.config.get("prompt", {}).get(key)
                for key in (
                    "prompt_prefix", "prompt_suffix", "negative_prompt", "translate_to_english",
                    "translation_provider_id", "translation_template", "translation_max_output_chars",
                )
            },
            "preview_max_chars": self._int_cfg("plan", "preview_translation_max_output_chars", 4000),
        }
        return hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _validated_plan_reference_paths(self, session: dict[str, Any]) -> list[str]:
        paths = [os.path.abspath(str(path)) for path in session.get("reference_image_paths", []) if path]
        if len(paths) > self._max_reference_images() or any(not os.path.isfile(path) for path in paths):
            raise RuntimeError("已选参考图已失效或超出数量限制，请重新附图或引用后重新建立计划；本次不会减少参考图或改为文生图")
        prepared = self._prepare_reference_image_paths(paths)
        if prepared != paths:
            raise RuntimeError("部分已选参考图不可用，请重新附图或引用后重新建立计划；本次不会减少参考图")
        return paths

    async def _prepare_plan_submission(self, session: dict[str, Any]) -> dict[str, Any]:
        lock = session.setdefault("submission_lock", asyncio.Lock())
        async with lock:
            if session.get("planning"):
                raise RuntimeError("当前计划正在整理，请稍候再试")
            render_options = self._freeze_plan_render_options(session)
            fingerprint = self._plan_submission_fingerprint(session)
            prepared = session.get("prepared_submission")
            if isinstance(prepared, dict) and prepared.get("fingerprint") == fingerprint:
                return prepared
            session.pop("prepared_submission", None)
            prompt = self._plan_final_prompt(session) or self._plan_raw_prompt(session)
            cn_prompt = self._build_prompt(prompt, (session.get("options") or {}).get("style"))
            final_prompt = await self._plan_mixed_prompt_preview(cn_prompt)
            if not final_prompt or not final_prompt.strip():
                raise RuntimeError("最终提示词为空，请检查计划或翻译模型后重试")
            if self._plan_submission_fingerprint(session) != fingerprint:
                raise RuntimeError("准备期间计划或生图模型已修改，请查看最新计划后重试")
            connection = self._resolve_image_api_connection(advanced=True)
            prepared = {
                "fingerprint": fingerprint,
                "chinese_prompt": cn_prompt,
                "final_prompt": final_prompt,
                "image_selection": (connection["provider_id"], connection["model"], connection["base_url"]),
                "render_options": render_options,
            }
            session["prepared_submission"] = prepared
            return prepared

    def _fallback_plan_response(self, session: dict[str, Any], *, error: str = "") -> str:
        raw_prompt = self._plan_raw_prompt(session)
        refs = session.get("reference_image_paths") or []
        opts = session.get("options") or {}
        final_prompt = self._build_prompt(raw_prompt, opts.get("style"))
        lines = []
        if error:
            lines.extend([f"（Plan 模型调用失败，先使用本地整理兜底：{error}）", ""])
        lines.extend(
            [
                "【图像需求】",
                raw_prompt or "（还没有文字需求）",
                "",
                "【参考图使用方式】",
                f"参考图：{len(refs)} 张",
                "",
                "【最终提示词】",
                final_prompt or raw_prompt or "（还没有最终提示词）",
                "",
                "【可修改项】",
                "继续用 /plan 追加或修改主体、动作、构图、风格、尺寸、质量和参考图关系。",
            ]
        )
        return "\n".join(lines).strip()

    def _format_plan_session_status(self, session: dict[str, Any] | None) -> str:
        if not session:
            return self._plan_help_text()
        raw_prompt = self._plan_raw_prompt(session)
        refs = session.get("reference_image_paths") or []
        opts = session.get("options") or {}
        planned_response = str(session.get("planned_response") or "").strip()
        selection = (session.get("prepared_submission") or {}).get("image_selection") or ()
        model_text = f"{selection[0]}（{selection[1]}）" if len(selection) >= 2 else "尚未准备"
        if planned_response:
            lines = [
                "GPT Image · 龙绘 Plan 已整理。",
                f"所属：{self._plan_owner_text(session)}",
                f"生图模型：{model_text}",
                "",
                "计划整理稿（不代表最终提交文本）：\n" + planned_response,
                "",
                "最终提交提示词：\n" + str((session.get("prepared_submission") or {}).get("final_prompt") or "尚未准备"),
                "",
                f"参考图：{len(refs)} 张",
            ]
            option_text = self._format_plan_options(opts)
            if option_text:
                lines.append(f"参数：{option_text}")
            if session.get("planning_error"):
                lines.append(f"提示：Plan 模型上次调用失败，当前展示为兜底整理。错误：{session.get('planning_error')}")
            lines.extend(
                [
                    "",
                    "继续修改：/plan 追加描述",
                    "确认生成：/plan confirm",
                    "退出：/plan quit",
                ]
            )
            return "\n".join(lines).strip()
        lines = [
            "GPT Image · 龙绘 Plan 已记录。",
            f"所属：{self._plan_owner_text(session)}",
            "",
            "当前需求：",
            raw_prompt or "（还没有文字需求）",
            "",
            f"参考图：{len(refs)} 张",
        ]
        option_text = self._format_plan_options(opts)
        if option_text:
            lines.extend(["参数：", option_text])
        lines.extend(
            [
                "",
                "继续补充：/plan 追加描述",
                "确认生成：/plan confirm",
                "退出：/plan quit",
            ]
        )
        return "\n".join(lines)

    def _build_plan_session_chain(self, event: AstrMessageEvent, session: dict[str, Any]) -> list[Any]:
        planned_response = str(session.get("planned_response") or "").strip()
        if not planned_response:
            return [Plain(self._format_plan_session_status(session))]

        refs = session.get("reference_image_paths") or []
        opts = session.get("options") or {}
        params = self._format_plan_options(opts) or "默认"
        selection = (session.get("prepared_submission") or {}).get("image_selection") or ()
        model_text = f"{selection[0]}（{selection[1]}）" if len(selection) >= 2 else "尚未准备"
        status_text = (
            "GPT Image · 龙绘 Plan 已整理。"
            f"\n所属：{self._plan_owner_text(session, event)}"
            f"\n生图模型：{model_text}"
            f"\n参考图：{len(refs)} 张"
            f"\n参数：{params}"
        )
        if session.get("planning_error"):
            status_text += f"\n提示：Plan 模型上次调用失败，当前展示为兜底整理。\n错误：{session.get('planning_error')}"

        node_items: list[tuple[str, str]] = [("Plan 状态", status_text)]
        node_items.extend(
            ("计划提示词草稿" if "最终提示词" in title else title, body)
            for title, body in self._split_plan_response_sections(planned_response)
        )
        prepared = session.get("prepared_submission") or {}
        node_items.append(("最终提交提示词", str(prepared.get("final_prompt") or "尚未准备")))
        node_items.append(
            (
                "操作",
                "继续修改：/plan 追加描述\n确认生成：/plan confirm\n退出：/plan quit",
            )
        )

        try:
            nodes = Nodes([])
            self_id = str(event.get_self_id() or "0")
            for title, text in node_items:
                chunks = self._chunk_plan_node_text(text)
                total = len(chunks)
                for index, chunk in enumerate(chunks, start=1):
                    node_title = title if total == 1 else f"{title} {index}/{total}"
                    nodes.nodes.append(
                        Node(
                            uin=self_id,
                            name=PLUGIN_DISPLAY_NAME,
                            content=[Plain(f"{node_title}\n{chunk}")],
                        )
                    )
            return [nodes]
        except Exception:
            logger.debug("GPT Image · 龙绘 failed to build merged plan status nodes", exc_info=True)
            return [Plain(self._format_plan_session_status(session))]

    def _split_plan_response_sections(self, planned_response: str) -> list[tuple[str, str]]:
        text = str(planned_response or "").strip()
        if not text:
            return []

        sections: list[tuple[str, str]] = []
        for match in re.finditer(r"【([^】]+)】\s*(.*?)(?=\n\s*【[^】]+】|\Z)", text, flags=re.S):
            title = match.group(1).strip()
            body = match.group(2).strip()
            if title and body:
                sections.append((title, body))
        if sections:
            return sections

        return [("整理稿", text)]

    def _chunk_plan_node_text(self, text: str, limit: int = 1600) -> list[str]:
        source = str(text or "").strip()
        if not source:
            return ["（空）"]
        if len(source) <= limit:
            return [source]

        chunks: list[str] = []
        remaining = source
        while len(remaining) > limit:
            split_at = max(
                remaining.rfind("\n\n", 0, limit),
                remaining.rfind("\n", 0, limit),
                remaining.rfind("。", 0, limit),
                remaining.rfind("，", 0, limit),
            )
            if split_at < max(300, limit // 3):
                split_at = limit
            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks or ["（空）"]

    def _format_plan_options(self, opts: dict[str, Any]) -> str:
        labels = {
            "output_format": "格式",
            "frame_count": "帧数",
            "frame_duration_ms": "每帧时长",
            "size": "尺寸",
            "aspect_ratio": "比例",
            "resolution": "清晰度",
            "quality": "质量",
            "style": "风格",
            "transparent_background": "透明背景",
        }
        parts: list[str] = []
        for key in ("output_format", "frame_count", "frame_duration_ms", "size", "aspect_ratio", "resolution", "quality", "style", "transparent_background"):
            value = opts.get(key)
            if value is None or value == "" or value is False:
                continue
            if key == "output_format":
                value = str(value).upper()
            elif key == "frame_count":
                value = f"{value} 帧"
            elif key == "frame_duration_ms":
                value = f"{value} 毫秒"
            parts.append(f"{labels.get(key, key)}={value}")
        return "，".join(parts)

    async def _plan_mixed_prompt_preview(self, cn_prompt: str) -> str:
        if not self._bool_cfg("prompt", "translate_to_english", False):
            return cn_prompt
        preview_max_chars = max(
            2000,
            self._int_cfg("prompt", "translation_max_output_chars", 4000),
            self._int_cfg("plan", "preview_translation_max_output_chars", 4000),
        )
        return await self._translate_prompt_to_english(
            cn_prompt,
            max_output_chars=preview_max_chars,
        )

    def _build_plan_confirm_chain(
        self,
        event: AstrMessageEvent,
        *,
        trace: dict[str, Any],
        raw_prompt: str,
        cn_prompt: str,
        mixed_prompt: str,
        options: dict[str, Any],
        reference_count: int,
        owner_text: str = "",
    ) -> list[Any]:
        flow_id = self._trace_flow_id(trace) or "未知"
        params = self._format_plan_options(options) or "默认"
        operation = "改图/参考图生成" if reference_count else "文生图"
        node_items = [
            ("流程ID", f"流程ID：{flow_id}\n任务类型：{operation}\n所属：{owner_text or '未知用户'}"),
            ("中文需求摘要", raw_prompt),
            ("翻译前提示词", cn_prompt),
            ("最终提交提示词", mixed_prompt),
            ("参数", f"参数：{params}\n参考图：{reference_count} 张"),
        ]
        try:
            nodes = Nodes([])
            self_id = str(event.get_self_id() or "0")
            for title, text in node_items:
                chunks = self._chunk_plan_node_text(text)
                total = len(chunks)
                for index, chunk in enumerate(chunks, start=1):
                    node_title = title if total == 1 else f"{title} {index}/{total}"
                    nodes.nodes.append(
                        Node(
                            uin=self_id,
                            name=PLUGIN_DISPLAY_NAME,
                            content=[Plain(f"{node_title}\n{chunk}")],
                        )
                    )
            return [nodes]
        except Exception:
            logger.debug("GPT Image · 龙绘 failed to build merged plan nodes", exc_info=True)
            lines = ["高级画图计划已确认"]
            for title, text in node_items:
                lines.extend(["", f"【{title}】", text])
            return [Plain("\n".join(lines))]

    def _new_flow_id(self) -> str:
        return f"gi-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def _trace_flow_id(self, trace: dict[str, Any] | None) -> str | None:
        if not trace:
            return None
        flow_id = trace.get("flow_id")
        return str(flow_id) if flow_id else None

    def _new_image_trace(
        self,
        event: AstrMessageEvent,
        *,
        trigger: str,
        operation: str,
        raw_user_prompt: str,
        prompt: str,
        reference_image_paths: list[str] | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        now = time.time()
        flow_id = self._new_flow_id()
        trace: dict[str, Any] = {
            "version": 1,
            "flow_id": flow_id,
            "status": "accepted",
            "created_at": now,
            "created_at_text": self._trace_time_text(now),
            "updated_at": now,
            "updated_at_text": self._trace_time_text(now),
            "trigger": trigger,
            "operation": operation,
            "event": self._trace_event_info(event),
            "raw_user_prompt": raw_user_prompt,
            "initial_prompt": prompt,
            "reference_images": [os.path.abspath(path) for path in (reference_image_paths or [])],
            "options": options or {},
            "stages": [],
            "result": {},
            "error": None,
        }
        self._trace_stage(
            trace,
            "accepted",
            status="accepted",
            raw_user_prompt=raw_user_prompt,
            initial_prompt=prompt,
        )
        return trace

    def _trace_route_decision(
        self,
        trace: dict[str, Any] | None,
        *,
        route: str,
        reason: str,
        operation: str,
        reference_image_paths: list[str] | None = None,
        **extra: Any,
    ) -> None:
        reference_image_paths = reference_image_paths or []
        decision: dict[str, Any] = {
            "route": route,
            "reason": reason,
            "operation": operation,
            "used_reference_images": bool(reference_image_paths),
            "reference_count": len(reference_image_paths),
        }
        decision.update(extra)
        self._trace_update(trace, route_decision=decision)
        self._trace_stage(trace, "route_decision", status="succeeded", **decision)

    def _trace_stage(self, trace: dict[str, Any] | None, name: str, **data: Any) -> None:
        if trace is None:
            return
        now = time.time()
        stage = {
            "name": name,
            "at": now,
            "at_text": self._trace_time_text(now),
        }
        stage.update(data)
        trace.setdefault("stages", []).append(self._trace_jsonable(stage))
        self._trace_update(trace)

    def _trace_update(self, trace: dict[str, Any] | None, **fields: Any) -> None:
        if trace is None:
            return
        for key, value in fields.items():
            trace[key] = self._trace_jsonable(value)
        now = time.time()
        trace["updated_at"] = now
        trace["updated_at_text"] = self._trace_time_text(now)
        self._save_image_trace(trace)

    def _trace_fail(self, trace: dict[str, Any] | None, exc: Exception) -> None:
        if trace is None:
            return
        error = self._friendly_error(exc)
        status = "delivery_failed" if trace.get("status") == "succeeded" and trace.get("result") else "failed"
        self._trace_update(trace, status=status, error=error)

    def _trace_time_text(self, timestamp: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    def _trace_access_identity(self, event: AstrMessageEvent) -> dict[str, str] | None:
        """Use AstrBot's platform-isolated session and authenticated sender identity."""
        try:
            identity = {
                "platform_id": str(event.get_platform_id() or "").strip(),
                "unified_msg_origin": str(event.unified_msg_origin or "").strip(),
                "sender_id": str(event.get_sender_id() or "").strip(),
            }
        except Exception:
            return None
        return identity if all(identity.values()) else None

    def _can_access_image_trace(self, event: AstrMessageEvent, trace: dict[str, Any]) -> bool:
        try:
            # AstrMessageEvent.is_admin() checks the role set by WakingCheckStage;
            # is_private_chat() checks the framework's FRIEND_MESSAGE type.
            # Never disclose another conversation's content into a group.
            if event.is_admin() is True and event.is_private_chat() is True:
                return True
        except Exception:
            pass
        stored = trace.get("event")
        current = self._trace_access_identity(event)
        if not isinstance(stored, dict) or current is None:
            return False
        if not isinstance(stored.get("sender_id"), (str, int)):
            return False
        if str(stored["sender_id"]) != current["sender_id"]:
            return False
        if "platform_id" in stored or "unified_msg_origin" in stored:
            return all(
                isinstance(stored.get(key), str) and stored[key] == current[key]
                for key in ("platform_id", "unified_msg_origin")
            )
        # Older traces stored the full UMO as the first composite-scope token.
        # Later tokens include bare session IDs that can themselves resemble a
        # different platform's UMO, so they must never authorize a query.
        scope = stored.get("scope")
        origin = current["unified_msg_origin"]
        return (
            isinstance(scope, str)
            and origin.startswith(current["platform_id"] + ":")
            and scope.split("|", 1)[0] == origin
        )

    def _trace_event_info(self, event: AstrMessageEvent) -> dict[str, Any]:
        info: dict[str, Any] = {
            "scope": self._event_scope_key(event),
            "message_key": self._event_message_key(event),
            "message_text": self._event_text(event),
        }
        message_obj = getattr(event, "message_obj", None)
        for name in ("message_id", "msg_id", "group_id", "sender_id", "user_id", "session_id"):
            value = getattr(message_obj, name, None)
            if value:
                info[name] = value
        for name in ("get_group_id", "get_sender_id", "get_session_id"):
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                logger.debug("Failed to read AstrBot event info for image trace", exc_info=True)
                continue
            if value:
                info[name[4:]] = value
        identity = self._trace_access_identity(event)
        if identity:
            info.update(identity)
        return self._trace_jsonable(info)

    def _trace_response_metadata(self, response: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if isinstance(response, dict):
            headers = response.get("_http_response_headers")
            if isinstance(headers, dict):
                metadata["headers"] = {str(key): str(value) for key, value in headers.items()}
        return self._trace_jsonable(metadata)

    def _trace_api_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        ignored_keys = {"image", "images", "file", "files"}
        return {key: value for key, value in payload.items() if key not in ignored_keys}

    def _trace_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseException):
            return self._friendly_error(value)
        if isinstance(value, dict):
            return {str(key): self._trace_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._trace_jsonable(item) for item in value]
        return str(value)

    def _trace_path(self, flow_id: str) -> Path:
        safe_id = self._normalize_flow_id(flow_id)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", safe_id)
        return self._trace_dir / f"{safe_id}.json"

    def _normalize_flow_id(self, value: str) -> str:
        text = (value or "").strip()
        match = re.search(r"gi-\d{8}-\d{6}-[0-9a-fA-F]{8}", text)
        if match:
            return match.group(0)
        return text

    def _save_image_trace(self, trace: dict[str, Any]) -> None:
        flow_id = self._trace_flow_id(trace)
        if not flow_id:
            return
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            path = self._trace_path(flow_id)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(self._trace_jsonable(trace), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception:
            logger.warning("Failed to save GPT Image trace %s", flow_id, exc_info=True)

    def _load_image_trace(self, flow_id: str) -> dict[str, Any] | None:
        normalized = self._normalize_flow_id(flow_id)
        if not normalized:
            return None
        path = self._trace_path(normalized)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load GPT Image trace %s", normalized, exc_info=True)
            return None
        return data if isinstance(data, dict) else None

    def _build_trace_result_chain(self, trace: dict[str, Any]) -> list[Any]:
        chain: list[Any] = [Plain(self._format_trace_summary(trace))]
        image_path = ((trace.get("result") or {}) if isinstance(trace.get("result"), dict) else {}).get("image_path")
        if image_path and os.path.exists(str(image_path)):
            chain.append(Image.fromFileSystem(str(image_path)))
        return chain

    def _trace_query_prompt(self, trace: dict[str, Any]) -> tuple[str, str]:
        """Prefer the payload paired with a successful API attempt, not a preview."""
        pending: dict[str, str] = {}
        successful_prompt = ""
        last_request_prompt = ""
        stages = trace.get("stages") if isinstance(trace.get("stages"), list) else []
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("name") != "image_api":
                continue
            profile = str(stage.get("profile") or "")
            payload = stage.get("payload")
            prompt = payload.get("prompt") if isinstance(payload, dict) else None
            if stage.get("status") == "started" and isinstance(prompt, str) and prompt:
                pending[profile] = prompt
                last_request_prompt = prompt
            elif stage.get("status") == "failed":
                pending.pop(profile, None)
            elif stage.get("status") == "succeeded":
                value = prompt if isinstance(prompt, str) and prompt else pending.get(profile, "")
                if value:
                    successful_prompt = value
        if successful_prompt:
            return "最终提交提示词（接口成功）", successful_prompt
        final_prompt = trace.get("final_prompt")
        if isinstance(final_prompt, str) and final_prompt:
            return "最终提示词（记录值，未核对成功请求）", final_prompt
        if last_request_prompt:
            return "请求提示词（未确认接口成功）", last_request_prompt
        return "最终提示词", "尚未产生最终提示词。"

    def _format_trace_summary(self, trace: dict[str, Any]) -> str:
        stages = trace.get("stages") if isinstance(trace.get("stages"), list) else []
        api = trace.get("api") if isinstance(trace.get("api"), dict) else {}
        lines = [
            f"流程ID：{self._clip_trace_text(trace.get('flow_id', ''), limit=80)}",
            f"状态：{self._trace_status_label(str(trace.get('status', '')))}；类型：{self._trace_operation_label(str(trace.get('operation', '')))}",
            f"入口：{self._clip_trace_text(trace.get('trigger', ''), limit=40)}；模型：{self._clip_trace_text(api.get('model', ''), limit=80) or '未记录'}",
            f"创建时间：{self._clip_trace_text(trace.get('created_at_text', ''), limit=32)}",
        ]
        label, final_prompt = self._trace_query_prompt(trace)
        # Reserve the first and largest part of the reply for the actual prompt.
        # Other stages are summarized below instead of repeating every full text.
        prompt_limit = min(TRACE_PROMPT_PREVIEW_LIMIT, max(200, TRACE_QUERY_TEXT_LIMIT - len("\n".join(lines)) - 160))
        lines.append(f"{label}：\n{final_prompt[:prompt_limit]}")
        if len(final_prompt) > prompt_limit:
            lines.append(f"……提示词已截断（完整 {len(final_prompt)} 字）；完整记录由管理员在 traces 中查看。")
        raw_prompt = str(trace.get("raw_user_prompt") or "").strip()
        if raw_prompt and raw_prompt != final_prompt:
            lines.append(f"原始需求：{self._clip_trace_text(raw_prompt, limit=160)}")
        stage_summary = " → ".join(
            f"{stage.get('name', '')}:{stage.get('status', '')}"
            for stage in stages if isinstance(stage, dict)
        )
        if stage_summary:
            lines.append(f"处理阶段：{self._clip_trace_text(stage_summary, limit=180)}")
        references = trace.get("reference_images")
        if isinstance(references, list) and references:
            lines.append(f"参考图：{len(references)} 张")
        api_stage = self._last_trace_stage(stages, ("image_api",))
        error = trace.get("error") or ((api_stage or {}).get("error") if (api_stage or {}).get("status") == "failed" else "")
        if error:
            lines.append(f"错误：{self._clip_trace_text(error, limit=180)}")
        text = "\n".join(lines)
        if len(text) <= TRACE_QUERY_TEXT_LIMIT:
            return text
        notice = "\n……摘要已截断；完整记录由管理员在 traces 中查看。"
        return text[:TRACE_QUERY_TEXT_LIMIT - len(notice)].rstrip() + notice

    def _last_trace_stage(self, stages: list[Any], names: tuple[str, ...]) -> dict[str, Any] | None:
        for stage in reversed(stages):
            if isinstance(stage, dict) and stage.get("name") in names:
                return stage
        return None

    def _trace_status_label(self, status: str) -> str:
        labels = {
            "accepted": "已接收",
            "running": "处理中",
            "succeeded": "成功",
            "failed": "失败",
            "delivery_failed": "已生成但发送失败",
        }
        return labels.get(status, status or "未知")

    def _trace_operation_label(self, operation: str) -> str:
        if operation == "edit":
            return "改图"
        if operation == "generation":
            return "生图"
        return operation or "未知"

    def _clip_trace_text(self, value: Any, *, limit: int = TRACE_PROMPT_PREVIEW_LIMIT) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _tool_request_key(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        size: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        style: str,
        transparent_background: bool,
        reference_paths: list[str],
        render_options: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "scope": self._event_scope_key(event),
            "prompt": prompt.strip(),
            "size": (size or "").strip(),
            "aspect_ratio": (aspect_ratio or "").strip(),
            "resolution": (resolution or "").strip(),
            "quality": (quality or "").strip(),
            "style": (style or "").strip(),
            "transparent_background": bool(transparent_background),
            "reference_paths": [str(Path(path).resolve()) for path in reference_paths],
            "render_options": render_options or self._resolve_render_options(
                size=size, aspect_ratio=aspect_ratio, resolution=resolution,
                reference_image_paths=reference_paths,
            ),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _tool_turn_key(self, event: AstrMessageEvent) -> str:
        payload = {
            "scope": self._event_scope_key(event),
            "message": self._event_message_key(event),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _event_message_key(self, event: AstrMessageEvent) -> str:
        parts: list[str] = []
        message_obj = getattr(event, "message_obj", None)
        for name in ("message_id", "msg_id", "id", "seq"):
            value = getattr(message_obj, name, None)
            if value:
                parts.append(str(value))

        for name in ("get_message_id", "get_msg_id"):
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                logger.debug("Failed to read AstrBot message id", exc_info=True)
                continue
            if value:
                parts.append(str(value))

        text = ""
        try:
            text = str(getattr(event, "message_str", "") or event.get_message_str() or "")
        except Exception:
            logger.debug("Failed to read AstrBot message text for image tool key", exc_info=True)
            text = str(getattr(event, "message_str", "") or "")
        if text:
            parts.append(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])

        return "|".join(dict.fromkeys(parts)) or "unknown-message"

    def _event_identity_value(
        self,
        event: AstrMessageEvent,
        attr_names: tuple[str, ...],
        method_names: tuple[str, ...],
    ) -> str:
        message_obj = getattr(event, "message_obj", None)
        for name in attr_names:
            value = getattr(message_obj, name, None)
            if value:
                return str(value)

        for container in (getattr(message_obj, "sender", None), getattr(event, "sender", None)):
            if not container:
                continue
            if isinstance(container, dict):
                for name in (*attr_names, "user_id", "sender_id", "id", "qq", "uin"):
                    value = container.get(name)
                    if value:
                        return str(value)
            else:
                for name in (*attr_names, "user_id", "sender_id", "id", "qq", "uin"):
                    value = getattr(container, name, None)
                    if value:
                        return str(value)

        for name in attr_names:
            value = getattr(event, name, None)
            if value:
                return str(value)

        for name in method_names:
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                logger.debug("Failed to read AstrBot event identity value %s", name, exc_info=True)
                continue
            if value:
                return str(value)
        return ""

    def _event_scope_key(self, event: AstrMessageEvent) -> str:
        parts: list[str] = []
        for name in ("unified_msg_origin", "session_id"):
            value = getattr(event, name, None)
            if value:
                parts.append(str(value))

        message_obj = getattr(event, "message_obj", None)
        for name in ("session_id", "group_id", "sender_id", "user_id"):
            value = getattr(message_obj, name, None)
            if value:
                parts.append(str(value))

        for name in ("get_session_id", "get_group_id", "get_sender_id"):
            method = getattr(event, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                logger.debug("Failed to read AstrBot event scope", exc_info=True)
                continue
            if value:
                parts.append(str(value))

        return "|".join(dict.fromkeys(parts)) or "global"

    def _message_allows_image_tool(self, event: AstrMessageEvent, *, use_reference_images: bool) -> bool:
        text = self._event_text(event)
        if not text:
            return False

        normalized = re.sub(r"\s+", "", text.lower())
        if not normalized:
            return False

        prompt_only_patterns = [
            r"(翻译|译成|翻成|翻译成|translate|translation).*(中文|英文|日文|日语|韩文|韩语|法文|德文|俄文|to|into)",
            r"(这段话|这段|这句话|这句|这段文字|这段英文|这段日文|这段prompt|thistext|thissentence).*(翻译|译成|翻成|translate)",
            r"^(翻译|译成|翻成|translate).+",
            r"(提示词|prompt).*(写|改|优化|润色|翻译|扩写|整理)",
            r"(生成|写|改|优化|润色|翻译|扩写|整理).*(提示词|prompt)",
            r"(怎么|如何|为什么|什么|啥|能不能|可以吗|参数|设置|教程|建议|方案|分析|评价).*(画|生成|生图|出图|改图|图片|图像)",
            r"(how|what|why|help|guide|tutorial|advice|suggest|analy[sz]e|review).*(draw|generate|create|edit|image|picture|photo|prompt)",
            r"(prompt).*(write|improve|rewrite|translate|optimi[sz]e|polish)",
            r"(write|improve|rewrite|translate|optimi[sz]e|polish).*(prompt)",
        ]
        if any(re.search(pattern, normalized) for pattern in prompt_only_patterns):
            return False
        # The LLM has already chosen this tool. Do not re-run the old keyword
        # router here, otherwise natural language requests can be blocked by a
        # stale regex even when the model routed correctly.
        return True

    def _event_text(self, event: AstrMessageEvent) -> str:
        try:
            return str(getattr(event, "message_str", "") or event.get_message_str() or "").strip()
        except Exception:
            logger.debug("Failed to read AstrBot message text", exc_info=True)
            return str(getattr(event, "message_str", "") or "").strip()

    def _parse_inline_options(self, text: str) -> tuple[dict[str, Any], str]:
        opts: dict[str, Any] = {
            "size": "",
            "aspect_ratio": "",
            "resolution": "",
            "quality": "",
            "style": "",
            "transparent_background": False,
            "output_format": "",
            "frame_count": None,
            "frame_duration_ms": None,
        }
        if not text:
            return opts, ""

        try:
            shlex.split(text)
            token_spans = list(re.finditer(r'''(?:[^\s"'\\]|\\.|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')+''', text))
            tokens = [shlex.split(match.group())[0] for match in token_spans]
        except (ValueError, IndexError):
            token_spans = list(re.finditer(r"\S+", text))
            tokens = [match.group() for match in token_spans]

        prompt_parts: list[str] = []
        prompt_indexes: list[int] = []
        invalid_option = False
        value_flags = {
            "--size", "-s", "--ratio", "--aspect", "--aspect-ratio", "--比例",
            "--resolution", "--res", "--分辨率", "--quality", "-q", "--style",
            "--format", "--frames", "--frame-ms",
        }
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token_spans[i].group().startswith(("\"", "'", "\\")):
                prompt_indexes.append(i)
                i += 1
                continue
            if token in value_flags and (
                i + 1 >= len(tokens) or token_spans[i + 1].group().startswith("--")
            ):
                invalid_option = True
                i += 1
                continue
            gif_flags = {"--format": "output_format", "--frames": "frame_count", "--frame-ms": "frame_duration_ms"}
            flag, separator, inline_value = token.partition("=")
            if flag in gif_flags:
                has_following_value = i + 1 < len(tokens) and not tokens[i + 1].startswith("--")
                value = inline_value if separator else tokens[i + 1] if has_following_value else ""
                if not value.strip():
                    invalid_option = True
                opts[gif_flags[flag]] = value if value.strip() else "__missing__"
                i += 1 if separator or not has_following_value else 2
                continue
            if token in {"--size", "-s"} and i + 1 < len(tokens):
                self._assign_size_like_option(opts, tokens[i + 1])
                i += 2
                continue
            if token.startswith("--size="):
                self._assign_size_like_option(opts, token.split("=", 1)[1])
                i += 1
                continue
            if token in {"--ratio", "--aspect", "--aspect-ratio", "--比例"} and i + 1 < len(tokens):
                opts["aspect_ratio"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith(("--ratio=", "--aspect=", "--aspect-ratio=", "--比例=")):
                opts["aspect_ratio"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--resolution", "--res", "--分辨率"} and i + 1 < len(tokens):
                opts["resolution"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith(("--resolution=", "--res=", "--分辨率=")):
                opts["resolution"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--quality", "-q"} and i + 1 < len(tokens):
                opts["quality"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith("--quality="):
                opts["quality"] = token.split("=", 1)[1]
                i += 1
                continue
            if token == "--style" and i + 1 < len(tokens):
                opts["style"] = tokens[i + 1]
                i += 2
                continue
            if token.startswith("--style="):
                opts["style"] = token.split("=", 1)[1]
                i += 1
                continue
            if token in {"--transparent", "--transparent-background", "--no-bg"}:
                opts["transparent_background"] = True
                i += 1
                continue
            prompt_indexes.append(i)
            i += 1

        previous = None
        for index in prompt_indexes:
            if previous is not None:
                prompt_parts.append(text[token_spans[previous].end():token_spans[index].start()] if index == previous + 1 else " ")
            prompt_parts.append(token_spans[index].group())
            previous = index
        if invalid_option:
            opts["output_format"] = "__missing__"
        return opts, "".join(prompt_parts).strip()

    def _assign_size_like_option(self, opts: dict[str, Any], value: str) -> None:
        normalized = (value or "").strip().lower()
        if normalized == "auto":
            opts["size"] = "auto"
            return
        if re.fullmatch(r"\d+[x:：]\d+", normalized):
            if ":" in normalized or "：" in normalized:
                opts["aspect_ratio"] = normalized.replace("：", ":")
            else:
                opts["size"] = normalized
            return
        if normalized in VALID_RESOLUTIONS:
            opts["resolution"] = normalized
            return
        opts["size"] = value

    def _extract_command_prompt(self, event: AstrMessageEvent, command_names: tuple[str, ...]) -> str:
        text = self._event_text(event)
        for name in command_names:
            for prefix in (f"/{name}", name):
                if text == prefix:
                    return ""
                if text.startswith(prefix):
                    return text[len(prefix) :].lstrip()
        return text

    async def _collect_reference_image_paths(
        self,
        event: AstrMessageEvent,
        *,
        include_recent: bool = False,
        stabilize: bool = False,
        source_records: list[dict[str, str]] | None = None,
    ) -> list[str]:
        # Retained only for compatibility with older callers. A previous chat
        # image is never evidence that the user selected it for this request.
        del include_recent
        if source_records is not None:
            source_records.clear()
        max_images = self._max_reference_images()
        paths: list[str] = []
        seen: set[str] = set()

        async def add_image(component: Image, source: str) -> None:
            if len(paths) >= max_images:
                return
            try:
                image_path = await component.convert_to_file_path()
            except Exception:
                logger.warning("Failed to convert AstrBot image component to file path", exc_info=True)
                return
            normalized = os.path.abspath(image_path)
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
                if source_records is not None:
                    source_records.append({"path": normalized, "source": source})

        async def walk(chain: Any, source: str = "current_message") -> None:
            if not chain or len(paths) >= max_images:
                return
            if isinstance(chain, Image):
                await add_image(chain, source)
                return
            if not isinstance(chain, (list, tuple)):
                return

            for component in chain:
                if len(paths) >= max_images:
                    break
                component_source = "quoted_message" if isinstance(component, Reply) else source
                if isinstance(component, Image):
                    await add_image(component, component_source)

                nested_chain = getattr(component, "chain", None)
                if nested_chain:
                    await walk(nested_chain, component_source)

                nested_content = getattr(component, "content", None)
                if nested_content:
                    await walk(nested_content, component_source)

                nested_nodes = getattr(component, "nodes", None)
                if nested_nodes:
                    for node in nested_nodes:
                        await walk(getattr(node, "content", None), component_source)

        try:
            message_chain = event.get_messages()
        except Exception:
            logger.debug("Failed to read AstrBot message chain", exc_info=True)
            message_chain = getattr(getattr(event, "message_obj", None), "message", [])

        await walk(message_chain)
        if len(paths) < max_images:
            await self._add_quoted_reference_image_paths(
                event, paths, seen, max_images, source_records=source_records,
            )
        if stabilize and paths:
            paths = await self._stabilize_reference_image_paths(paths, source_records=source_records)
        return paths

    async def _add_quoted_reference_image_paths(
        self,
        event: AstrMessageEvent,
        paths: list[str],
        seen: set[str],
        max_images: int,
        *,
        source_records: list[dict[str, str]] | None = None,
    ) -> None:
        try:
            quoted_image_refs = await extract_quoted_message_images(event)
        except Exception:
            logger.warning("Failed to extract images from quoted message", exc_info=True)
            return

        added = 0
        for image_ref in quoted_image_refs:
            if len(paths) >= max_images:
                break
            try:
                image_path = await self._materialize_reference_image_ref(image_ref)
            except Exception:
                logger.warning(
                    "Failed to materialize quoted reference image: %s",
                    str(image_ref)[:160],
                    exc_info=True,
                )
                continue

            normalized = os.path.abspath(image_path)
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
                if source_records is not None:
                    source_records.append({"path": normalized, "source": "quoted_message"})
                added += 1

        if quoted_image_refs:
            logger.info(
                "GPT Image quoted reference fallback refs=%d added=%d total=%d",
                len(quoted_image_refs),
                added,
                len(paths),
            )

    async def _materialize_reference_image_ref(self, image_ref: str) -> str:
        value = (image_ref or "").strip()
        if not value:
            raise RuntimeError("empty quoted image ref")

        lower_value = value.lower()
        if lower_value.startswith("base64://"):
            raw_b64 = value[len("base64://") :].strip()
            try:
                image_bytes = base64.b64decode(raw_b64)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError("quoted image base64 cannot be decoded") from exc
            if not image_bytes:
                raise RuntimeError("quoted image base64 is empty")
            if len(image_bytes) > MAX_REFERENCE_IMAGE_BYTES:
                raise RuntimeError("quoted image exceeds 50MB")
            suffix = self._image_suffix_from_bytes(image_bytes) or ".png"
            file_path = self._image_dir / (
                f"gpt_image_ref_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
            )
            file_path.write_bytes(image_bytes)
            return str(file_path)

        if lower_value.startswith(("http://", "https://")):
            file_path = self._image_dir / (
                f"gpt_image_ref_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            )
            return str(await self._download_image(value, file_path))

        if lower_value.startswith("file://"):
            file_path = value[7:]
            if file_path.startswith("/") and len(file_path) > 3 and file_path[2] == ":":
                file_path = file_path[1:]
            value = file_path

        if os.path.exists(value):
            return os.path.abspath(value)

        raise RuntimeError("quoted image ref is not a resolvable file or URL")

    async def _stabilize_reference_image_paths(
        self, paths: list[str], *, source_records: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Copy event-owned references into the plugin cache before background work."""

        prepared = self._prepare_reference_image_paths(paths)
        stable_paths: list[str] = []
        sources_by_path = {
            os.path.abspath(record["path"]): record.get("source", "explicit_selection")
            for record in (source_records or []) if record.get("path")
        }
        stable_sources: list[dict[str, str]] = []
        seen_digests: set[str] = set()
        copied = 0
        reused = 0
        deduplicated = 0
        image_dir = self._image_dir.resolve()

        for image_path in prepared:
            source = Path(image_path).resolve()
            digest = await asyncio.to_thread(self._reference_file_sha256, source)
            if digest in seen_digests:
                deduplicated += 1
                continue
            seen_digests.add(digest)

            try:
                source.relative_to(image_dir)
                target = source
                reused += 1
            except ValueError:
                content_type = self._detect_image_content_type(str(source))
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }[content_type]
                target = image_dir / f"gpt_image_ref_{digest}{suffix}"
                if not target.exists():
                    temporary_target = image_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
                    try:
                        await asyncio.to_thread(shutil.copy2, source, temporary_target)
                        os.replace(temporary_target, target)
                    finally:
                        temporary_target.unlink(missing_ok=True)
                    copied += 1
                else:
                    target.touch()
                    reused += 1

            stable_paths.append(str(target))
            if source_records is not None:
                stable_sources.append({
                    "path": str(target),
                    "source": sources_by_path.get(os.path.abspath(image_path), "explicit_selection"),
                })

        logger.info(
            "GPT Image stabilized reference images count=%d copied=%d reused=%d deduplicated=%d",
            len(stable_paths),
            copied,
            reused,
            deduplicated,
        )
        if source_records is not None:
            source_records[:] = stable_sources
        return stable_paths

    @staticmethod
    def _reference_file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _prepare_reference_image_paths(self, paths: list[str]) -> list[str]:
        prepared: list[str] = []
        seen: set[str] = set()
        rejected: list[str] = []
        for image_path in paths[: self._max_reference_images()]:
            path = os.path.abspath(image_path)
            if path in seen:
                continue
            seen.add(path)
            name = Path(path).name or "unnamed"
            if not os.path.isfile(path):
                rejected.append(f"{name}:missing")
                continue
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                rejected.append(f"{name}:{type(exc).__name__}")
                continue
            if size > MAX_REFERENCE_IMAGE_BYTES:
                rejected.append(f"{name}:too_large")
                continue
            content_type = self._detect_image_content_type(path)
            if content_type not in VALID_INPUT_CONTENT_TYPES:
                rejected.append(f"{name}:unsupported")
                continue
            prepared.append(path)

        if paths and not prepared:
            details = ", ".join(rejected[:3]) or "unavailable"
            raise RuntimeError(f"没有可用参考图（{details}），请重新发送或引用图片后再试。")
        if rejected:
            logger.warning(
                "GPT Image skipped unusable reference images count=%d details=%s",
                len(rejected),
                ", ".join(rejected[:5]),
            )
        return prepared

    def _max_reference_images(self) -> int:
        configured = self._int_cfg("image", "max_reference_images", MAX_GPT_IMAGE_REFERENCE_IMAGES)
        return min(MAX_GPT_IMAGE_REFERENCE_IMAGES, max(1, configured))

    def _detect_image_content_type(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as file_obj:
                header = file_obj.read(16)
        except OSError:
            logger.debug("Failed to read image header for content type detection", exc_info=True)
            header = b""

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"

        guessed, _ = mimetypes.guess_type(image_path)
        return guessed or "application/octet-stream"

    def _multipart_filename(self, image_path: str, content_type: str) -> str:
        path = Path(image_path)
        filename = path.name or f"reference_{uuid.uuid4().hex[:8]}"
        if path.suffix:
            return filename

        extension = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }.get(content_type, "")
        return f"{filename}{extension}"

    def _normalize_openai_size(
        self,
        size: str | None,
        *,
        aspect_ratio: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        explicit_size = self._normalize_pixel_size(size)
        if explicit_size:
            return explicit_size

        explicit_ratio = self._parse_aspect_ratio(aspect_ratio) or self._parse_aspect_ratio(size)
        if explicit_ratio:
            return self._openai_size_for_ratio(explicit_ratio)

        if prefer_reference_ratio and reference_image_paths:
            reference_size = self._reference_image_upstream_size(reference_image_paths[0])
            if reference_size:
                return reference_size

        configured_size = self._normalize_pixel_size(self._str_cfg("image", "size", "1024x1024"))
        if configured_size:
            return configured_size

        configured_ratio = self._parse_aspect_ratio(self._str_cfg("image", "aspect_ratio", "auto"))
        if configured_ratio:
            return self._openai_size_for_ratio(configured_ratio)

        return "1024x1024"

    def _normalize_pixel_size(self, size: str | None) -> str:
        value = (size or "").strip().lower()
        if not value or value == "auto":
            return ""
        value = value.replace("*", "x").replace(" ", "")
        if value in VALID_UPSTREAM_IMAGE_SIZES:
            return value
        return ""

    def _normalize_aspect_ratio(
        self,
        aspect_ratio: str | None,
        *,
        size: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        # Priority order is explicit request, explicit size, reference image,
        # plugin defaults, then auto. This keeps image editing close to the
        # original picture unless the user or bot asks for a new ratio.
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio)
        if explicit_ratio:
            return explicit_ratio

        size_ratio = self._parse_aspect_ratio(size)
        if size_ratio:
            return size_ratio

        size_value = self._normalize_pixel_size(size)
        if size_value in OPENAI_SIZE_TO_RATIO:
            return OPENAI_SIZE_TO_RATIO[size_value]

        if prefer_reference_ratio and reference_image_paths:
            reference_ratio = self._reference_image_aspect_ratio(reference_image_paths[0])
            if reference_ratio:
                return reference_ratio

        configured_ratio = self._parse_aspect_ratio(self._str_cfg("image", "aspect_ratio", "auto"))
        if configured_ratio:
            return configured_ratio

        configured_size = self._str_cfg("image", "size", "")
        configured_size_ratio = self._parse_aspect_ratio(configured_size)
        if configured_size_ratio:
            return configured_size_ratio
        configured_pixel_size = self._normalize_pixel_size(configured_size)
        if configured_pixel_size in OPENAI_SIZE_TO_RATIO:
            return OPENAI_SIZE_TO_RATIO[configured_pixel_size]

        return "auto"

    def _parse_aspect_ratio(self, value: str | None) -> str:
        text = (value or "").strip().lower().replace("：", ":")
        if not text:
            return ""
        if text == "auto":
            return ""
        match = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
        if not match:
            return ""
        left = int(match.group(1))
        right = int(match.group(2))
        if left <= 0 or right <= 0:
            return ""
        ratio = f"{left}:{right}"
        if ratio in VALID_ASPECT_RATIOS:
            return ratio
        return self._nearest_supported_aspect_ratio(left / right)

    def _reference_image_aspect_ratio(self, image_path: str) -> str:
        size = self._image_dimensions(image_path)
        if not size:
            return ""
        width, height = size
        if width <= 0 or height <= 0:
            return ""
        return self._nearest_supported_aspect_ratio(width / height)

    def _reference_image_upstream_size(self, image_path: str) -> str:
        size = self._image_dimensions(image_path)
        if not size:
            return ""
        width, height = size
        if width <= 0 or height <= 0:
            return ""

        ratio = width / height
        if ratio >= REFERENCE_SQUARE_RATIO_MAX:
            return REFERENCE_LANDSCAPE_SIZE
        if ratio <= REFERENCE_SQUARE_RATIO_MIN:
            return REFERENCE_PORTRAIT_SIZE
        return REFERENCE_SQUARE_SIZE

    def _nearest_supported_aspect_ratio(self, ratio: float) -> str:
        candidates = [item for item in VALID_ASPECT_RATIOS if item != "auto"]

        def ratio_value(item: str) -> float:
            left, right = item.split(":", 1)
            return int(left) / int(right)

        return min(candidates, key=lambda item: abs(ratio_value(item) - ratio))

    def _openai_size_for_ratio(self, ratio: str) -> str:
        if ratio == "1:1":
            return "1024x1024"
        left, right = ratio.split(":", 1)
        ratio_value = int(left) / int(right)
        if ratio_value > 1:
            return "1536x1024"
        return "1024x1536"

    def _normalize_compatible_size(
        self,
        size: str | None,
        *,
        aspect_ratio: str | None,
        reference_image_paths: list[str],
        prefer_reference_ratio: bool,
    ) -> str:
        explicit_size = self._normalize_pixel_size(size)
        if explicit_size:
            return explicit_size

        explicit_ratio = self._parse_aspect_ratio(size)
        if explicit_ratio:
            return explicit_ratio

        ratio = self._normalize_aspect_ratio(
            aspect_ratio,
            size=None,
            reference_image_paths=reference_image_paths,
            prefer_reference_ratio=prefer_reference_ratio,
        )
        if ratio != "auto":
            return ratio

        configured_size = self._str_cfg("image", "size", "")
        configured_pixel_size = self._normalize_pixel_size(configured_size)
        if configured_pixel_size:
            return configured_pixel_size
        configured_ratio = self._parse_aspect_ratio(configured_size)
        if configured_ratio:
            return configured_ratio
        return "auto"

    def _normalize_resolution(self, resolution: str | None) -> str:
        value = (resolution or "").strip().lower()
        if not value:
            value = self._str_cfg("image", "resolution", "auto").lower()
        value = value.replace(" ", "")
        if value in {"1", "1k", "1024", "1024p"}:
            return "1k"
        if value in {"2", "2k", "2048", "2048p"}:
            return "2k"
        return value if value in VALID_RESOLUTIONS else "auto"

    def _image_dimensions(self, image_path: str) -> tuple[int, int] | None:
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as image:
                return image.size
        except (ImportError, OSError, ValueError):
            logger.debug("Failed to read image dimensions with Pillow", exc_info=True)

        try:
            with open(image_path, "rb") as file_obj:
                data = file_obj.read(256 * 1024)
        except OSError:
            logger.debug("Failed to read image header for dimension detection", exc_info=True)
            return None

        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")

        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return self._webp_dimensions(data)

        if data.startswith(b"\xff\xd8"):
            return self._jpeg_dimensions(data)

        return None

    def _jpeg_dimensions(self, data: bytes) -> tuple[int, int] | None:
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > len(data):
                return None
            segment_length = int.from_bytes(data[i : i + 2], "big")
            if segment_length < 2 or i + segment_length > len(data):
                return None
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                height = int.from_bytes(data[i + 3 : i + 5], "big")
                width = int.from_bytes(data[i + 5 : i + 7], "big")
                return width, height
            i += segment_length
        return None

    def _webp_dimensions(self, data: bytes) -> tuple[int, int] | None:
        chunk_type = data[12:16]
        if chunk_type == b"VP8X" and len(data) >= 30:
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk_type == b"VP8 " and len(data) >= 30:
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return width, height
        if chunk_type == b"VP8L" and len(data) >= 25:
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            width = 1 + (((b1 & 0x3F) << 8) | b0)
            height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return width, height
        return None

    def _apply_load_saving_options(
        self,
        *,
        operation: str,
        raw_user_prompt: str,
        size: str | None,
        resolution: str | None,
        quality: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Drop model-suggested heavy params unless the user asked for them."""

        prompt = (raw_user_prompt or "").lower()
        explicit_heavy = bool(
            re.search(r"\b[1248]\s*k\b", prompt)
            or re.search(r"\d{3,4}\s*[x×]\s*\d{3,4}", prompt)
            or any(
                token in prompt
                for token in (
                    "高清",
                    "高质量",
                    "高分辨率",
                    "精修",
                    "精细",
                    "细节拉满",
                    "最终稿",
                    "原图尺寸",
                    "大图",
                )
            )
        )
        if explicit_heavy:
            return size, resolution, quality

        normalized_size = self._normalize_pixel_size(size)
        if operation == "edit" and normalized_size in {
            "2560x1440",
            "1440x2560",
            "2880x2880",
            "3840x2160",
            "2160x3840",
        }:
            size = None

        if (resolution or "").strip().lower() == "2k":
            resolution = None
        if (quality or "").strip().lower() == "high":
            quality = None
        return size, resolution, quality

    def _normalize_quality(self, quality: str | None) -> str:
        value = (quality or "").strip().lower()
        if not value:
            value = self._str_cfg("image", "quality", "auto").lower()
        return value if value in VALID_QUALITIES else "auto"

    def _output_format(self) -> str:
        value = self._str_cfg("image", "output_format", "png").lower()
        return value if value in VALID_OUTPUT_FORMATS else "png"

    def _resolve_render_options(
        self,
        *,
        output_format: str | None = None,
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
        size: str | None = None,
        reference_image_paths: list[str] | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
    ) -> dict[str, Any]:
        """Validate and freeze the format and native GIF controls before awaits.

        Args:
            output_format: Explicit output format, or the configured default.
            frame_count: Explicit GIF frame count, or the configured default.
            frame_duration_ms: Explicit GIF frame duration, or its default.
            size: Exact GIF pixel dimensions, or the configured size.
            reference_image_paths: Selected references, forbidden for GIF.
            aspect_ratio: Explicit ratio; GIF accepts only empty or auto.
            resolution: Explicit resolution tier; GIF accepts only empty or auto.

        Returns:
            A per-job snapshot of the normalized rendering settings.

        Raises:
            RuntimeError: If the requested settings cannot be honored.
        """
        selected_format = (output_format or self._str_cfg("image", "output_format", "png")).strip().lower()
        if selected_format not in VALID_OUTPUT_FORMATS:
            raise RuntimeError("输出格式只能为 png、jpeg、webp 或 gif。")
        if selected_format != "gif" and (frame_count is not None or frame_duration_ms is not None):
            raise RuntimeError("帧数和每帧时长仅适用于 GIF 动图；请指定 --format gif，或移除帧参数。")
        options: dict[str, Any] = {"output_format": selected_format}
        if selected_format == "gif":
            if not self._bool_cfg("gif", "enabled", False):
                raise RuntimeError("GIF 功能未开启。请在插件配置的“GIF 动图”中启用，并确认使用支持原生 GIF 的官方 API 源及完整透传参数的中转。")
            if reference_image_paths:
                raise RuntimeError("本插件的 GIF 动图目前仅支持纯文字生成，不支持参考图或改图；本次不会丢弃参考图或改成静态图片。")
            if any(str(value or "").strip().lower() not in {"", "auto"} for value in (aspect_ratio, resolution)):
                raise RuntimeError("GIF 不支持显式比例或清晰度档位，请移除 --ratio/--resolution，改用 --size 指定宽高均为 16 倍数的像素尺寸，例如 512x512。")
            settings = self._section("gif")
            values = {
                "frame_count": frame_count if frame_count is not None else settings.get("frame_count", 9),
                "frame_duration_ms": frame_duration_ms if frame_duration_ms is not None else settings.get("frame_duration_ms", 200),
            }
            for key, value in values.items():
                if isinstance(value, bool) or not (
                    isinstance(value, int) or isinstance(value, str) and re.fullmatch(r"\d+", value)
                ):
                    raise RuntimeError("GIF 帧数和每帧时长必须为整数。")
                number = int(value)
                minimum, maximum = (2, 16) if key == "frame_count" else (20, 2000)
                if not minimum <= number <= maximum:
                    raise RuntimeError("GIF 帧数须为 2–16，每帧时长须为 20–2000 毫秒。")
                options[key] = (number + 5) // 10 * 10 if key == "frame_duration_ms" else number
            gif_size = (size or self._str_cfg("gif", "size", "auto") or "auto").strip().lower()
            gif_size = gif_size.replace("×", "x").replace("*", "x").replace(" ", "")
            if gif_size != "auto":
                match = re.fullmatch(r"(\d+)x(\d+)", gif_size)
                if not match or any(int(side) <= 0 or int(side) % 16 for side in match.groups()):
                    raise RuntimeError("GIF 像素尺寸须为 auto，或宽和高均为 16 的正整数倍，例如 512x512。")
                gif_size = "x".join(str(int(side)) for side in match.groups())
            options["size"] = gif_size
        return options

    def _images_generation_url(self, base_url: str | None = None) -> str:
        base_url = (base_url if base_url is not None else self._base_url()).rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/images/edits"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/generations"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/generations"
        return f"{base_url}/v1/images/generations"

    def _images_edit_url(self, base_url: str | None = None) -> str:
        base_url = (base_url if base_url is not None else self._base_url()).rstrip("/")
        if base_url.endswith("/images/edits"):
            return base_url
        if base_url.endswith("/images/generations"):
            return f"{base_url.rsplit('/images/', 1)[0]}/images/edits"
        if base_url.endswith("/v1"):
            return f"{base_url}/images/edits"
        return f"{base_url}/v1/images/edits"

    def _model_name(self, *, advanced: bool = False) -> str:
        return self._resolve_image_api_connection(advanced=advanced)["model"]

    def _base_url(self, *, advanced: bool = False) -> str:
        return self._resolve_image_api_connection(advanced=advanced)["base_url"]

    def _timeout_seconds(self) -> int:
        return self._int_cfg("api", "timeout_seconds", DEFAULT_API_TIMEOUT)

    def _base_url_prefers_standard_profile(self, base_url: str | None = None) -> bool:
        base_url = (base_url if base_url is not None else self._base_url()).lower()
        standard_markers = (
            "ruoli.dev",
            "localhost:3000",
            "127.0.0.1:3000",
        )
        return any(marker in base_url for marker in standard_markers)

    def _auto_detect_profile(
        self,
        *,
        operation: str,
        size: str | None,
        aspect_ratio: str | None,
        reference_image_paths: list[str] | None,
        base_url: str | None = None,
    ) -> str:
        """Pick an initial compat profile when the cache misses.

        The returned value is only a first guess; if the upstream rejects
        the request with a parameter-format error the caller will retry once
        with the other profile.
        """
        if self._base_url_prefers_standard_profile(base_url):
            return "standard"

        # An explicit pixel size unambiguously speaks the strict OpenAI
        # dialect, so try ``standard`` first.
        if self._normalize_pixel_size(size):
            return "standard"

        # OpenAI strict can only express 1:1, 3:2, and 2:3 cleanly via its
        # three accepted pixel sizes. Any other ratio (16:9, 21:9, ...) is
        # better expressed by keeping the ratio in ``size`` (flexible). The
        # retry mechanism still falls back to standard if flexible fails.
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio) or self._parse_aspect_ratio(size)
        if explicit_ratio and explicit_ratio not in {"1:1", "3:2", "2:3"}:
            return "flexible"

        # Image edits with reference images and no explicit ratio: flexible
        # can keep the original ratio without rounding to one of three sizes.
        if operation == "edit" and reference_image_paths:
            return "flexible"

        return "standard"

    def _request_has_strong_profile_hint(
        self,
        *,
        operation: str,
        size: str | None,
        aspect_ratio: str | None,
        reference_image_paths: list[str] | None,
        base_url: str | None = None,
    ) -> bool:
        """Whether this request should override the cached compat profile.

        Cache is useful for ambiguous defaults, but an explicit pixel size,
        a wide/tall ratio that OpenAI strict cannot represent exactly, or an
        edit request that should follow the reference image is a stronger
        signal than whatever succeeded on an earlier request.
        """
        if self._base_url_prefers_standard_profile(base_url):
            return True
        if self._normalize_pixel_size(size):
            return True
        explicit_ratio = self._parse_aspect_ratio(aspect_ratio) or self._parse_aspect_ratio(size)
        if explicit_ratio and explicit_ratio not in {"1:1", "3:2", "2:3"}:
            return True
        return operation == "edit" and bool(reference_image_paths)

    def _build_api_error(self, status: int, body: str) -> ImageAPIError:
        try:
            data = json.loads(body)
            error = data.get("error", data) if isinstance(data, dict) else data
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
        except (json.JSONDecodeError, AttributeError, TypeError):
            message = body
        return ImageAPIError(status, str(message)[:600], body)

    def _extract_image_response(self, data: Any) -> dict[str, Any] | None:
        if isinstance(data, dict):
            payload_data = data.get("data")
            if isinstance(payload_data, list) and payload_data:
                # Normalize alias keys (``image_url`` / ``base64``) used by
                # webpage-reverse / 2api-style providers so the saver only
                # has to look at ``url`` / ``b64_json``.
                normalized_items: list[dict[str, Any]] = []
                for item in payload_data:
                    if not isinstance(item, dict):
                        continue
                    url = self._normalize_response_url(item.get("url") or item.get("image_url"))
                    b64 = item.get("b64_json") or item.get("base64")
                    if url or b64:
                        normalized_items.append({
                            "url": url,
                            "b64_json": b64,
                            "revised_prompt": item.get("revised_prompt"),
                        })
                if normalized_items:
                    return {"data": normalized_items}

            nested = data.get("result") or data.get("output")
            if nested is not None:
                nested_response = self._extract_image_response(nested)
                if nested_response:
                    return nested_response

            for key in ("results", "images", "urls", "image_urls", "result_images"):
                nested_images = data.get(key)
                if nested_images is not None:
                    nested_response = self._extract_image_response(nested_images)
                    if nested_response:
                        return nested_response

            if isinstance(payload_data, dict):
                nested_response = self._extract_image_response(payload_data)
                if nested_response:
                    return nested_response

            url = self._normalize_response_url(data.get("url") or data.get("image_url"))
            if url:
                return {"data": [{"url": url, "revised_prompt": data.get("revised_prompt")}]}
            b64_json = data.get("b64_json") or data.get("base64")
            if isinstance(b64_json, str) and b64_json.strip():
                return {"data": [{"b64_json": b64_json, "revised_prompt": data.get("revised_prompt")}]}

        if isinstance(data, list):
            image_items = []
            for item in data:
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    image_items.append({"url": item})
                elif isinstance(item, dict) and (
                    item.get("url") or item.get("b64_json") or item.get("image_url") or item.get("base64")
                ):
                    url = self._normalize_response_url(item.get("url") or item.get("image_url"))
                    b64 = item.get("b64_json") or item.get("base64")
                    if url or b64:
                        image_items.append(
                            {
                                "url": url,
                                "b64_json": b64,
                                "revised_prompt": item.get("revised_prompt"),
                            }
                        )
            if image_items:
                return {"data": image_items}

        return None

    def _normalize_response_url(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
        return ""

    def _image_suffix_from_bytes(self, data: bytes, content_type: str = "") -> str:
        content_type = (content_type or "").split(";", 1)[0].strip().lower()
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if content_type == "image/png" or data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if content_type in {"image/jpeg", "image/jpg"} or data.startswith(b"\xff\xd8"):
            return ".jpg"
        if content_type == "image/webp" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
            return ".webp"
        return ""

    def _validate_generated_gif(self, data: bytes, *, render_options: dict[str, Any] | None = None) -> None:
        """Reject a static or corrupt upstream response without altering bytes.

        Args:
            data: Original bytes returned by the image provider.
            render_options: Frozen requested animation settings, when available.

        Raises:
            RuntimeError: If the response is not a decodable multi-frame GIF.
        """
        if not data.startswith((b"GIF87a", b"GIF89a")):
            raise RuntimeError("请求的是 GIF 动图，但供应商返回了其他格式；本次不会发送静态图片代替。")
        try:
            from io import BytesIO
            from PIL import Image as PILImage

            with PILImage.open(BytesIO(data)) as animation:
                frame_total = getattr(animation, "n_frames", 1)
                if animation.format != "GIF" or not 2 <= frame_total <= 16:
                    raise ValueError("not animated")
                durations = []
                for frame in range(frame_total):
                    animation.seek(frame)
                    animation.load()
                    durations.append(animation.info.get("duration", 0))
        except ImportError:
            raise RuntimeError("当前环境缺少 GIF 校验支持，已停止发送，请管理员检查 Pillow 依赖。") from None
        except Exception:
            raise RuntimeError("供应商没有返回有效的 2–16 帧 GIF 动图；本次不会发送静态图或损坏文件代替。") from None
        if render_options and (
            frame_total != render_options["frame_count"]
            or any(duration != render_options["frame_duration_ms"] for duration in durations)
        ):
            raise RuntimeError(
                "供应商返回的 GIF 未按请求设置帧数或播放速度，已停止发送。"
                f"请求 {render_options['frame_count']} 帧、每帧 {render_options['frame_duration_ms']} 毫秒；"
                f"实际 {frame_total} 帧、每帧时长有 {sorted(set(durations))} 毫秒。"
            )

    def _resolve_image_api_connection(self, *, advanced: bool = False) -> dict[str, Any]:
        """Snapshot the selected provider connection without persisting secrets.

        Args:
            advanced: Select the independently configured Plan image model.

        Returns:
            A per-request connection containing the selected provider ID, model,
            base URL, current credential, and independent request headers.

        Raises:
            RuntimeError: If the selection or its Images API connection is not
                usable. Errors deliberately omit credentials and raw values.
        """
        selector = "advanced_provider_id" if advanced else "normal_provider_id"
        tier_name = "高级生图模型（/plan）" if advanced else "普通生图模型"
        provider_id = self._str_cfg("api", selector, "").strip()
        if not provider_id:
            raise RuntimeError(f"请先在插件的“模型选择”中选择 AstrBot 已配置的{tier_name}。")
        try:
            provider = self.context.get_provider_by_id(provider_id)
        except Exception:
            raise RuntimeError("无法读取所选生图供应商，请检查 AstrBot 供应商配置。") from None
        if provider is None:
            raise RuntimeError("所选生图供应商不存在、已停用或未加载，请重新选择已启用的供应商。")
        config = getattr(provider, "provider_config", None)
        if not isinstance(config, dict) or config.get("enable") is False:
            raise RuntimeError("所选生图供应商配置不可用或已停用，请重新选择。")
        if config.get("type") not in {"openai_chat_completion", "openai_responses"} or "api_version" in config:
            raise RuntimeError("请选择 OpenAI 兼容供应商，且该供应商须支持 Images 生图接口；暂不支持 Azure、Gemini 或 Anthropic 原生接口。")
        try:
            model = provider.get_model()
            api_key = provider.get_current_key()
            client = getattr(provider, "client", None)
            effective_base = getattr(client, "base_url", None)
            base_url = str(effective_base or config.get("api_base") or "").strip().rstrip("/")
        except Exception:
            raise RuntimeError("无法读取所选供应商的连接信息，请检查其地址和 API Key。") from None
        if not isinstance(model, str) or not model.strip() or any(ord(char) < 32 or ord(char) == 127 for char in model):
            raise RuntimeError("所选生图模型的模型名称无效，请在 AstrBot 模型设置中检查。")
        if not isinstance(api_key, str) or not api_key.strip() or any(ord(char) < 32 or ord(char) == 127 for char in api_key):
            raise RuntimeError("所选生图供应商没有有效的 API Key，请在 AstrBot 供应商设置中配置。")
        try:
            parsed = urlsplit(base_url)
            valid_url = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in base_url)
            )
            # Accessing port also validates malformed and out-of-range values.
            parsed.port
        except (TypeError, ValueError):
            valid_url = False
        if not valid_url:
            raise RuntimeError("所选生图供应商的 API 地址无效：须为 HTTP(S) 地址，且不能含账号、密码或查询参数。")
        headers = self._api_headers(api_key.strip(), custom_headers=config.get("custom_headers"))
        return {
            "provider_id": provider_id,
            "model": model.strip(),
            "base_url": base_url,
            "api_key": api_key.strip(),
            "headers": headers,
        }

    def _api_headers(
        self,
        api_key: str,
        *,
        content_type: str | None = None,
        custom_headers: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": self._user_agent(),
        }
        reserved = {"authorization", "content-type", "content-length", "host", "connection", "transfer-encoding", "proxy-authorization"}
        if isinstance(custom_headers, dict):
            for name, value in custom_headers.items():
                if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
                    raise RuntimeError("所选供应商的自定义请求头无效，请在 AstrBot 供应商设置中检查。")
                if name.lower() in reserved:
                    continue
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise RuntimeError("所选供应商的自定义请求头无效，请在 AstrBot 供应商设置中检查。")
                value = str(value)
                if any(ord(char) < 32 or ord(char) == 127 for char in value):
                    raise RuntimeError("所选供应商的自定义请求头无效，请在 AstrBot 供应商设置中检查。")
                if name.lower() == "user-agent":
                    if value.strip() and not self._str_cfg("api", "user_agent", "").strip():
                        headers["User-Agent"] = value
                else:
                    for old_name in list(headers):
                        if old_name.lower() == name.lower():
                            del headers[old_name]
                    headers[name] = value
        if content_type:
            headers["Content-Type"] = content_type

        organization = self._str_cfg("api", "organization", "")
        project = self._str_cfg("api", "project", "")
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        return headers

    def _user_agent(self) -> str:
        # Allow per-deployment override for middlemen that expect a specific
        # UA. Empty (the default) falls back to ``DEFAULT_USER_AGENT``.
        configured = self._str_cfg("api", "user_agent", "").strip()
        return configured or DEFAULT_USER_AGENT

    def _api_key(self, *, advanced: bool = False) -> str:
        return self._resolve_image_api_connection(advanced=advanced)["api_key"]

    def _ensure_enabled(self, *, advanced: bool = False) -> None:
        self._resolve_image_api_connection(advanced=advanced)

    def _cleanup_old_images(self) -> None:
        ttl_hours = self._int_cfg("runtime", "cache_ttl_hours", 72)
        max_files = self._int_cfg("runtime", "max_cache_files", 200)
        if ttl_hours <= 0 and max_files <= 0:
            return

        try:
            files = sorted(
                (path for path in self._image_dir.iterdir() if path.is_file()),
                key=lambda path: path.stat().st_mtime,
            )
            now = time.time()
            if ttl_hours > 0:
                expire_before = now - ttl_hours * 3600
                for path in files:
                    if path.stat().st_mtime < expire_before:
                        path.unlink(missing_ok=True)
                files = [path for path in files if path.exists()]
            if max_files > 0 and len(files) > max_files:
                for path in files[: len(files) - max_files]:
                    path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clean old GPT Image cache files", exc_info=True)

    def _format_api_error(self, status: int, text: str) -> str:
        cf_message = self._cf_challenge_message(status, text)
        if cf_message:
            return cf_message
        try:
            data = json.loads(text)
            error = data.get("error", data)
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
        except (json.JSONDecodeError, AttributeError, TypeError):
            message = text
        return f"图像接口请求失败：HTTP {status} {message[:600]}"

    def _cf_challenge_message(self, status: int, text: str) -> str | None:
        # Cloudflare's anti-bot/managed-challenge pages are returned as HTML
        # with a ``cf-ray`` marker. Surface them as a clear, actionable error
        # instead of dumping raw HTML to the user.
        if status not in {403, 503} or not text:
            return None
        sample = text[:4000]
        lowered = sample.lower()
        if (
            "cf-ray" not in lowered
            and "cf_ray" not in lowered
            and "ray id" not in lowered
            and "cloudflare" not in lowered
        ):
            return None
        if not any(
            marker in lowered
            for marker in (
                "<html",
                "<!doctype html",
                "challenge",
                "just a moment",
                "attention required",
                "安全验证",
                "验证页",
                "cf-ray",
                "cf_ray",
            )
        ):
            return None
        ray_match = _CF_RAY_PATTERN.search(sample)
        ray_part = f"，cf-ray={ray_match.group(1)}" if ray_match else ""
        return (
            f"图像接口被 Cloudflare 拦截 (HTTP {status}{ray_part})。"
            "上游中转启用了 anti-bot 防护：可在插件配置 api.user_agent 里换一个 UA，"
            "或联系中转方放行你的出口 IP。"
        )

    def _friendly_error(self, exc: Exception) -> str:
        text = str(exc).strip()
        return text if text else exc.__class__.__name__

    def _network_failure_message(self, exc: BaseException, action: str) -> str:
        # Timeout-specific message: bare ``TimeoutError`` tells the user nothing
        # actionable. Surface the actual configured timeout so they know which
        # knob to turn.
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return (
                f"{action}：等待上游响应 {self._timeout_seconds()} 秒后超时。"
                "4K 图像生成/改图本身较慢，可在配置里调高 api.timeout_seconds（建议 300-600），"
                "或检查中转/网络是否可达。"
            )
        return f"{action}：网络连接失败：{self._friendly_error(exc)}"

    def _section(self, section: str) -> dict[str, Any]:
        value = self.config.get(section, {}) if isinstance(self.config, dict) else {}
        return value if isinstance(value, dict) else {}

    def _str_cfg(self, section: str, key: str, default: str) -> str:
        value = self._section(section).get(key, default)
        if value is None:
            return default
        return str(value).strip()

    def _int_cfg(self, section: str, key: str, default: int) -> int:
        value = self._section(section).get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _bool_cfg(self, section: str, key: str, default: bool) -> bool:
        value = self._section(section).get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        return bool(value)
