# AstrBot GPT Image / Image2 Studio 定制版

这是基于 `starmiaoa/astrbot-plugin-gpt-image` 的定制版本，用于 AstrBot 中调用 GPT Image/OpenAI-compatible 图像生成与编辑接口。

本仓库只保存插件源码与开发文档，不保存任何运行配置、API Key、NapCat token、生成图片、数据库、日志或备份文件。

## 功能概览

- 支持 `/生图` 文生图。
- 支持 `/改图` 图生图/图片编辑，自动读取当前消息图片和回复/引用消息里的图片。
- 支持 LLM 工具调用 `generate_image_with_gpt_image_2`。
- 支持 OpenAI 官方 Images API、标准 OpenAI-compatible 接口，以及部分网页逆向/2api/ToAPIs 类渠道。
- 支持 `size`、`resolution`、`aspect_ratio` 的自动归一化。
- 支持提示词增强模型读取参考图并生成最终图像 prompt。
- 支持“自拍/人设生图”：用户让 bot 画“你、你的、自己、自拍、本体、形象”等内容时，可用插件内配置的人设改写最终 prompt。
- 支持引用图片或引用文字中的描述来生成新图，并可在同一会话内短时间沿用最近参考图。
- 支持在提交图片接口前把最终 prompt 交给文本模型英文化。
- 增加翻译类、提示词类请求的误触发拦截，减少把普通对话误判成图片任务。
- 支持 `Image2 Studio` 的 `/plan` 高级画图模式：先整理需求和最终 prompt，确认后再生成。
- 支持 `/生图流程`、`/查生图` 查询流程 ID 对应的路由、提示词、接口阶段和结果摘要。
- 支持图片接口 POST 短暂故障重试，并保留独立的下载重试配置。
- 支持扩展像素尺寸：`2560x1440`、`1440x2560`、`2880x2880`、`3840x2160`、`2160x3840`。
- 群聊中必须明确 @bot 才允许触发生图/改图，避免误触发和误扣额度。
- 群聊中 @bot 且文本明确是纯文字生图请求时，会直接进入图片生成流程，不再依赖对话模型自行决定是否调用工具。

## 主要定制点

### 1. 群聊必须 @bot

群聊里，以下入口都必须 @ 到 bot 本体才会继续执行：

- `/生图`
- `/改图`
- 自然语言直接触发的参考图改图
- 自然语言直接触发的纯文字生图
- LLM 函数工具 `generate_image_with_gpt_image_2`

私聊不受影响，可以直接发送生图或改图请求。

### 2. 直接参考图改图

当用户在同一条消息里发送图片或引用图片，并明确说“重绘、重画、改图、把图改成、参考这张图生成”等需求时，插件可以绕过对话模型，直接进入图片编辑流程。

这个流程会先做群聊 @bot 校验，避免未 @ 的群聊消息触发图片任务。

### 3. 提示词增强逻辑

`prompt_enhance.mode = placeholder` 时，当前逻辑已经简化：

- 不再使用旧的“主体内容提取提示词”先提取主体。
- 不再把提取结果塞进旧的“固定绘图母版”。
- 直接使用“完整提示词增强模板”生成最终 prompt。
- 如果模板里包含 `[在此处替换为您想要生成的主体内容]`，增强模型会被要求把它替换为用户本次真实需求。
- 如果增强模型仍然遗留占位符，插件会用用户原始请求兜底替换。

因此日常维护时，主要修改 WebUI 里的“完整提示词增强模板”。

### 4. 自拍/人设生图

当用户的纯文字生图请求包含“你、你的、自己、自拍、全身照、本体、形象”等指代时，插件会先把插件内配置的人设和用户请求交给文本模型改写成最终图像 prompt，再提交图片接口。

相关配置在 WebUI 的“自拍/人设生图”分组里：

- `self_persona.enable`：启用或关闭插件内生图人设。
- `self_persona.provider_id`：用于改写人设 prompt 的模型供应商。
- `self_persona.persona_name`：生图人设名称。
- `self_persona.persona_prompt`：固定视觉设定。
- `self_persona.fallback_to_astrbot_persona`：插件内人设为空时是否回退 AstrBot 当前人设。

### 5. 引用描述生图和最近参考图

用户引用一段文字、引用一张带有描述的图片，或先发图再短时间内发“按这个描述生成一张图”时，插件会调用提示词增强模型整理出最终生图 prompt。这个流程是纯文生图，不会强行保留参考图的构图和人物，除非描述本身提出这种要求。

`runtime.recent_reference_ttl_seconds` 控制最近参考图缓存时间，设为 `0` 可关闭。

### 6. 提示词英文化

`prompt.translate_to_english` 开启后，插件会在真正请求图片接口前，把已经整理好的最终 prompt 交给文本模型翻译/改写成英文。翻译模型、模板、超时和失败回退行为都在 `prompt` 分组中配置。

### 7. Image2 Studio Plan 模式

`/plan` 是多轮确认式画图流程：

- `/plan 画图需求`：创建或补充计划。
- `/plan status`：查看当前计划。
- `/plan confirm`：确认并开始生成。
- `/plan quit`：退出当前计划。

Plan 阶段可以读取参考图，调用配置的 Plan 模型整理“图像需求、参考图使用方式、最终提示词、可修改项”。确认前不会真正提交图片接口。

相关配置在 WebUI 的“高级画图 Plan 模式”分组里，包含 Plan 模型、最多参考图数、超时、输出长度、会话闲置超时和自定义整理模板。

### 8. 流程追踪

每次图像任务会生成一个流程 ID，并在任务接受回复中带出。可以用：

```text
/生图流程 流程ID
/查生图 流程ID
```

查询该任务的路由、提示词增强、英文化、图片接口、发送结果等阶段信息。流程文件保存在运行数据目录中，不应提交到本仓库。

### 9. 触发与重试策略

- `runtime.enable_direct_keyword_trigger` 控制自然语言关键词直接触发，默认关闭。关闭后，普通自然语言生图/改图主要交给对话模型工具路由；`/生图`、`/改图`、`/plan` 不受影响。
- LLM 工具调用前会二次校验消息是否具备明确图片意图，避免模型误调用工具后再回复“做好了”。
- `runtime.retry_times` 只用于下载图片等无副作用请求。
- `runtime.image_request_retry_times` 用于图片生成/修改 POST 请求在 502/503/504 或网络瞬断时重试；数值越大越稳，但也更可能重复生成或重复扣费。

### 10. 用户提交提示语

图片任务被接受后，插件固定回复：

```text
收到，开始画了，别催，如果等会提示报错喊老大排查
```

## 配置说明

在 AstrBot 插件配置里填写 API 信息：

- `api.api_key`：图像接口 API Key。不要提交到 Git。
- `api.base_url`：接口地址，通常填到 `/v1`。
- `api.model`：图像模型名，例如 `gpt-image-2`。
- `api.timeout_seconds`：图像任务超时时间，建议 120 到 300 秒。
- `image.size`：具体像素尺寸，例如 `1024x1024`、`1536x1024`、`3840x2160`。
- `image.resolution`：清晰度档位，目前保留 `auto`、`1k`、`2k`；需要 4K 时用 `image.size` 的 4K 像素尺寸。
- `image.aspect_ratio`：画面比例，例如 `1:1`、`16:9`、`9:16`。
- `prompt_enhance.enable`：是否启用提示词增强。
- `prompt_enhance.provider_id`：提示词增强模型供应商 ID。
- `prompt_enhance.use_global_image_caption_provider`：增强模型 Provider ID 为空时，优先使用 AstrBot 全局图片描述模型。
- `prompt_enhance.prompt_template`：完整提示词增强模板。
- `prompt.translate_to_english`：是否在上传图片接口前英文化 prompt。
- `self_persona.enable`：是否启用自拍/人设生图。
- `plan.enable_model_planning`：是否启用 Plan 模型整理。
- `runtime.enable_direct_keyword_trigger`：是否启用关键词直接触发。
- `runtime.image_request_retry_times`：图片接口 POST 重试次数。
- `runtime.recent_reference_ttl_seconds`：最近参考图有效时间。

配置文件位于 AstrBot 数据目录，不应进入本仓库。

## 常用命令

```text
/生图 一只赛博猫坐在霓虹窗边 --ratio 16:9 --resolution 2k --quality high
/生图 城市夜景海报 --size 3840x2160 --quality high
/改图 把背景换成夜晚城市，保留人物姿势 --resolution 2k
/改图 把这几张图合成一张电影海报 --ratio 2:3 --quality high
/plan 先帮我整理一个赛博风角色海报方案，确认后再画
/生图流程 gi-YYYYMMDD-HHMMSS-xxxxxxxx
```

群聊中需要：

```text
@bot /改图 把这张图重绘成 JOJO 画风，保留文字
```

私聊中可以直接：

```text
/改图 把这张图重绘成 JOJO 画风，保留文字
```

## 文件说明

- `main.py`：插件主逻辑。
- `_conf_schema.json`：AstrBot WebUI 配置 schema。
- `metadata.yaml`：AstrBot 插件元信息。
- `__init__.py`：插件包入口。
- `LICENSE`：原插件许可证。
- `DEVELOPMENT_SYNC.md`：开发、脱敏、同步、部署流程。
- `GPT_Image插件本地修改记录.md`：后续本地改动记录，作为同步 Git 的核对清单。

## 不要提交的内容

- `data/config/*_config.json`
- `data/plugin_data/`
- `main.py.bak*`
- `_conf_schema.json.bak*`
- `__pycache__/`
- 生成图片、日志、数据库
- 流程追踪数据、运行缓存、备份文件
- API Key、NapCat token、Tailscale 地址、内网 IP、个人 QQ 号等运行环境信息
