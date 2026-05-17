# AstrBot GPT Image 插件定制版

这是基于 `starmiaoa/astrbot-plugin-gpt-image` 的定制版本，用于 AstrBot 中调用 GPT Image/OpenAI-compatible 图像生成与编辑接口。

本仓库只保存插件源码与开发文档，不保存任何运行配置、API Key、NapCat token、生成图片、数据库、日志或备份文件。

## 功能概览

- 支持 `/生图` 文生图。
- 支持 `/改图` 图生图/图片编辑，自动读取当前消息图片和回复/引用消息里的图片。
- 支持 LLM 工具调用 `generate_image_with_gpt_image_2`。
- 支持 OpenAI 官方 Images API、标准 OpenAI-compatible 接口，以及部分网页逆向/2api/ToAPIs 类渠道。
- 支持 `size`、`resolution`、`aspect_ratio` 的自动归一化。
- 支持提示词增强模型读取参考图并生成最终图像 prompt。
- 群聊中必须明确 @bot 才允许触发生图/改图，避免误触发和误扣额度。

## 主要定制点

### 1. 群聊必须 @bot

群聊里，以下入口都必须 @ 到 bot 本体才会继续执行：

- `/生图`
- `/改图`
- 自然语言直接触发的参考图改图
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

### 4. 用户提交提示语

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
- `image.size`：具体像素尺寸，例如 `1024x1024`。
- `image.resolution`：清晰度档位，例如 `1k`、`2k`、`4k`。
- `image.aspect_ratio`：画面比例，例如 `1:1`、`16:9`、`9:16`。
- `prompt_enhance.enable`：是否启用提示词增强。
- `prompt_enhance.provider_id`：提示词增强模型供应商 ID。
- `prompt_enhance.prompt_template`：完整提示词增强模板。

配置文件位于 AstrBot 数据目录，不应进入本仓库。

## 常用命令

```text
/生图 一只赛博猫坐在霓虹窗边 --ratio 16:9 --resolution 2k --quality high
/改图 把背景换成夜晚城市，保留人物姿势 --resolution 2k
/改图 把这几张图合成一张电影海报 --ratio 2:3 --quality high
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

## 不要提交的内容

- `data/config/*_config.json`
- `data/plugin_data/`
- `main.py.bak*`
- `_conf_schema.json.bak*`
- `__pycache__/`
- 生成图片、日志、数据库
- API Key、NapCat token、Tailscale 地址、内网 IP、个人 QQ 号等运行环境信息

