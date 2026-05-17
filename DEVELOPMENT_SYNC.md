# 开发与同步文档

本文档用于后续从运行中的 AstrBot 机器同步插件改动、脱敏后提交到 Git，以及从 Git 部署回 bot 机器。

## 目录约定

运行中的插件目录：

```text
/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image
```

AstrBot 容器内插件目录：

```text
/AstrBot/data/plugins/astrbot_plugin_gpt_image
```

本仓库只对应这个插件目录中的发布文件：

```text
main.py
_conf_schema.json
metadata.yaml
__init__.py
LICENSE
README.md
DEVELOPMENT_SYNC.md
.gitignore
```

## 当前核心改造

### 群聊触发保护

相关函数：

- `_group_message_mentions_this_bot`
- `_message_chain_mentions_bot`
- `DirectReferenceImageEditFilter.filter`
- `generate_command`
- `edit_command`
- `generate_image_with_gpt_image_2`

规则：

1. 私聊直接放行。
2. 群聊必须检测到消息链里有 `At(qq=event.get_self_id())`。
3. 未 @bot 的群消息不允许触发生图/改图。
4. `@全体` 或 @其他人 不算。

### 提示词增强 placeholder 模式

相关函数：

- `_enhance_prompt_with_reference_images`
- `_prompt_enhance_placeholder`
- `_placeholder_fill_text`
- `_with_placeholder_fill_instruction`
- `_replace_prompt_placeholder`

当前设计：

1. `placeholder` 模式直接读取 `prompt_enhance.prompt_template`。
2. 增强模型根据完整模板输出最终 prompt。
3. 如果模板包含 `[在此处替换为您想要生成的主体内容]`，增强模型应把它替换为用户本次需求。
4. 如果模型输出仍包含占位符，插件用用户原始请求兜底替换。
5. `subject_prompt_template` 和 `final_prompt_template` 仅保留兼容旧配置，当前主流程不再使用。

### 直接参考图改图

相关入口：

- `DirectReferenceImageEditFilter`
- `direct_reference_image_edit_request`

用途：

当用户发送图片/引用图片并明确提出改图需求时，插件可以绕过对话模型直接启动图片任务，减少对话模型拦截导致的失败。

注意：

群聊里仍然必须 @bot。

## 从运行机器同步到本仓库

先从运行机器复制当前容器内插件文件，避免宿主挂载权限或旧文件问题：

```powershell
scp bot@<BOT_HOST>:/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image/main.py .\main.py
scp bot@<BOT_HOST>:/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image/_conf_schema.json .\_conf_schema.json
scp bot@<BOT_HOST>:/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image/metadata.yaml .\metadata.yaml
scp bot@<BOT_HOST>:/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image/__init__.py .\__init__.py
scp bot@<BOT_HOST>:/home/bot/qqbot/data/plugins/astrbot_plugin_gpt_image/LICENSE .\LICENSE
```

同步后检查敏感信息：

```powershell
rg -n "sk-[A-Za-z0-9_-]+|tvly-[A-Za-z0-9_-]+|token\s*[:=]|api[_-]?key\s*[:=]|Bearer\s+[A-Za-z0-9._-]+|192\.168\.|100\." .
```

允许出现变量名，例如 `api_key`、`Authorization: Bearer {api_key}`；不允许出现真实 key、token、内网地址或个人环境信息。

## 本地检查

```powershell
python -m py_compile .\main.py
Get-Content -Encoding UTF8 .\_conf_schema.json | ConvertFrom-Json | Out-Null
```

## 部署到运行机器

部署前备份：

```powershell
$ts = Get-Date -Format yyyyMMddHHmmss
ssh bot@<BOT_HOST> "docker cp astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/main.py /home/bot/astrbot_plugin_gpt_image_main.py.bak.$ts"
ssh bot@<BOT_HOST> "docker cp astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/_conf_schema.json /home/bot/astrbot_plugin_gpt_image_conf_schema.json.bak.$ts"
```

上传并写入容器：

```powershell
scp .\main.py bot@<BOT_HOST>:/home/bot/astrbot_plugin_gpt_image_main.py
scp .\_conf_schema.json bot@<BOT_HOST>:/home/bot/astrbot_plugin_gpt_image_conf_schema.json
ssh bot@<BOT_HOST> "docker cp /home/bot/astrbot_plugin_gpt_image_main.py astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/main.py"
ssh bot@<BOT_HOST> "docker cp /home/bot/astrbot_plugin_gpt_image_conf_schema.json astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/_conf_schema.json"
```

容器内语法检查：

```powershell
ssh bot@<BOT_HOST> "docker exec astrbot python3 -m py_compile /AstrBot/data/plugins/astrbot_plugin_gpt_image/main.py"
```

重启：

```powershell
ssh bot@<BOT_HOST> "docker restart astrbot"
```

只看插件加载和错误过滤日志：

```powershell
ssh bot@<BOT_HOST> "docker logs --since 2m astrbot 2>&1 | grep -E 'astrbot_plugin_gpt_image|GPT Image|Traceback|ERROR|ERRO|WARN' | grep -v 'core.event_bus' | tail -100"
```

## Git 提交流程

```bash
git status --short
git add main.py _conf_schema.json metadata.yaml __init__.py LICENSE README.md DEVELOPMENT_SYNC.md .gitignore
git commit -m "Customize GPT Image plugin for AstrBot deployment"
git push
```

提交前再次确认：

```bash
git diff --cached --check
git grep -n "sk-\|tvly-\|token\|Bearer \|192\.168\|100\."
```

如果 `git grep` 只命中变量名和文档里的检查命令，可以提交；如果命中真实 key、token、IP 或用户信息，必须先删除。

## 回滚

如果部署后插件异常：

```powershell
ssh bot@<BOT_HOST> "docker cp /home/bot/astrbot_plugin_gpt_image_main.py.bak.<时间戳> astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/main.py"
ssh bot@<BOT_HOST> "docker restart astrbot"
```

如果配置 schema 也回滚：

```powershell
ssh bot@<BOT_HOST> "docker cp /home/bot/astrbot_plugin_gpt_image_conf_schema.json.bak.<时间戳> astrbot:/AstrBot/data/plugins/astrbot_plugin_gpt_image/_conf_schema.json"
ssh bot@<BOT_HOST> "docker restart astrbot"
```

## 维护原则

- 不在仓库里保存任何运行配置和密钥。
- 插件逻辑改动尽量集中在 `main.py`，WebUI 配置字段说明同步改 `_conf_schema.json`。
- 先在测试群或私聊验证，再部署到长期运行的 bot。
- 对会产生费用的图片任务，优先加触发条件和去重保护。
- 任何会影响群聊自动触发的改动，都要再次验证“未 @ 不触发，@bot 才触发”。
