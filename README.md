# 内蒙古民族幼儿师范视频模型 · AI 视频创作工作台

这是在原有 **SGLang v0.5.22 MiniMax-H3 中文 Gradio WebUI** 上增加持久化生成记录的工作台。采用可切换的日间／夜间科技工作台布局，保留现有推理、任务、上传、输出与生成行为。

## 本次 UI 设计（v0.5.22-studio4.4 · 正式界面）

- **学校品牌标识**：左上角改为“内蒙古民族幼儿师范视频模型”，浏览器标题同步更新。
- **主题切换**：右上角日间／夜间模式按钮，实时更换背景、卡片、文本、上传控件等色彩，并在同一浏览器中记住选择。
- **三栏工作台**：左侧导航 / 中部创作与素材 / 右侧视频预览。
- **创作区**：大尺寸中文提示词输入、任务模式提示、首尾帧与参考素材上传。
- **参数区**：保留原有 7 项生成参数，收纳在可展开的高级面板中；**推理步数默认值由 50 调整为 2**，滑块范围 2–100。
- **预览区**：仍使用原生 Gradio `gr.Video`，支持原有含音频成片预览。
- **视觉**：深夜蓝／明亮云白双主题、紫蓝渐变、磨砂面板、自适应布局、统一控件样式。
- **科技动效**：动态网格、呼吸光效、流光按钮、环形渲染核心及波形动效。
- **生成等待舱**：点击原有生成按钮时显示科技 HUD、实时等待计时、无百分比的循环指示条；发现新视频输出或任务按钮恢复后关闭。可关闭光效或手动收起动画，操作不会取消模型推理。动效不代表后端真实阶段或进度。
- **无额外前端依赖**：CSS 内嵌在生成的 Python UI 文件中，仅增加原生 JavaScript 控制本地主题和展示等待动画，未引入新的 API、模型、JS 库、GPU 依赖或外部服务。

### V4.3 新增：跨重启生成记录

- 左侧增加“生成记录”入口；右侧可搜索、分页查看、选择、在线播放和下载原始 MP4（保留音频）。
- **不复制视频文件**：直接读取已核实为宿主机挂载的 `/data/minimax-h3/outputs/*.mp4`，原始视频仍留在原目录。
- 新增 `/data/minimax-h3/history/history.sqlite3`，记录 WebUI 任务开始、成功、失败和中断状态及参数；容器重建后保留历史。
- 页面加载及手动刷新会扫描旧 MP4 和同目录的 API 输出；无法证明来源的历史视频标记“自动导入”，不会伪造提示词等元数据。
- 扫描跳过近期仍在写入、空文件、无效 MP4 和越界符号链接；下载仅开放输出目录中的已有完成文件。
- **使用边界**：未对现有 18080 增加用户登录和个人隔离，能够访问该页面的人可以查看记录和下载视频。若需按员工分权限，必须另加认证。
- 视频文件需要由宿主机持续保留；若管理员清理 `outputs/`，数据库仍在，但相应记录无法播放或下载。

### 正式使用说明

实际工作台以部署后启动的 Gradio 页面为准，页面中不再显示“体验生成等待动效”和“不连接 GPU”等预览专用文案。点击原有“开始生成”会使用既有 SGLang 推理回调，并在等待期间呈现科技动画。**不能把 dev 目录下的单文件 HTML 当作视频生成服务**；仅修改文字不会建立 GPU 连接。

### 重要：保持原有推理功能

主题按钮只修改浏览器 CSS 显示状态与 localStorage；等待动画只监听生成按钮和视频结果在页面中的变化，不会向视频生成服务额外发送请求，也不会修改原有回调。

本次 **没有更改** `build_minimax_h3_sampling_params_kwargs`、`_generate_minimax_h3`、调度器请求、原生成回调 `generate` 的函数体、按钮提交参数与顺序、模型任务的来源和选择方式；生成按钮通过记录包装器调用原回调，除推理步数默认值调整外，也没有更改其他默认参数。`Ref2VA` 的“是否选择上传图片”开关、默认透明 PNG 回退和上传校验行为均保持原样。

此项目仅覆盖 SGLang Python 包中的 WebUI 文件 `minimax_h3.py` 并新增标准库实现的 `h3_studio_history.py` 和 `h3_studio_prompts.py`，无需修改模型权重、端口、Compose 主文件、CUDA 内核或推理后端。

## 下载后如何使用

### A. 从 GitHub 直接更新现有服务器（推荐，无需 Git）

以下命令在 **openEuler 主机**执行，安装前确保没有正在运行的视频生成任务：

```bash
curl -fL --retry 3 --connect-timeout 15 \
  https://raw.githubusercontent.com/Dugtx/SGLang-H3-ZH-WebUI/main/scripts/deploy_no_git.sh \
  -o /tmp/h3-deploy-no-git.sh
bash /tmp/h3-deploy-no-git.sh
```

部署脚本自动拉取 GitHub `main` 最新源码，核验 `v0.5.22-studio4.4`、重新生成覆盖文件后构建并重建容器。后端与权重、Docker Compose 主文件、端口及 GPU 参数不改变。实际生成能力仍需在运行服务器上验收。

### B. 使用本地 ZIP 解压源码部署（备用）

先把本 ZIP 解压后的整个 `SGLang-H3-ZH-WebUI-main` 目录放在服务器，然后在目录下执行：

```bash
python3 scripts/build_overlay.py
python3 -m unittest discover -s tests -v
bash scripts/deploy_no_git.sh
```

脚本将优先使用包含本版本 Studio 源码的本地目录。**两种部署方式都会重新创建 `sglang-h3` 容器，进行中的生成任务将中断。**

### C. 仅重新生成覆盖文件（不部署）

```bash
python3 scripts/build_overlay.py
python3 tests/test_localization.py
```

构建会根据 `upstream/minimax_h3.py`、`assets/workbench_layout.pyfrag`、`assets/workbench.css` 生成可直接覆盖的 `overlay/minimax_h3.py`、补丁 `patches/minimax_h3_zh.patch` 和 SHA-256 清单 `manifest.json`。上游版本或哈希不符合固定基线时直接报错。

## 界面预览与验证

- `dev/UI_设计预览_静态.html`：**静态设计示意**，无需服务器即可在浏览器查看界面和切换日间／夜间主题；不具有视频生成功能。页面已移除演示专用按钮与提示。
- `dev/preview_without_gpu.py`：开发时使用当前 Gradio 真正布局启动一个**不接入 GPU 的 UI 测试服务**，仅供检查外观和控件，生成按钮不能用于推理。
- `tests/test_localization.py` 和 `tests/test_waiting_ui.py`：不依赖 Gradio/GPU 的源代码完整性、动效与接口回归测试。

真实视频生成需要在已部署且正确配置的 MiniMax-H3 SGLang 服务上运行。静态预览或无 GPU 测试服务不代表 GPU 推理已完成验收。

## 目录

| 路径 | 内容 |
| --- | --- |
| `overlay/minimax_h3.py` | 生成后的自包含 UI 覆盖文件 |
| `assets/workbench_layout.pyfrag` | UI 工作台布局与历史记录面板模板 |
| `assets/history_store.py` | SQLite 历史索引、旧视频导入与路径保护 |
| `assets/workbench.css` | 深色／浅色视觉样式、悬浮态与响应式布局 |
| `assets/workbench.js` | 浏览器主题切换及等待 HUD 生命周期监控 |
| `assets/waiting_visual.html` | 内嵌在 Python UI 内的纯前端科技等待动画 |
| `assets/nm_mascot_header.png` | 左上角透明吉祥物品牌图 |
| `upstream/minimax_h3.py` | SHA-256 锁定的 SGLang 上游原版 |
| `scripts/build_overlay.py` | 确定性构建 + patch + manifest |
| `scripts/deploy_no_git.sh` | 本地 ZIP / GitHub 双模式部署 |
| `scripts/apply_overlay.py` | Docker 构建时校验覆盖文件 |
| `tests/test_localization.py` | 不改推理核心函数的离线检查 |
| `tests/test_history_store.py` | SQLite 重启持久化、真实 MP4、分页、搜索、路径保护测试 |
| `Dockerfile`、`compose.override.yaml` | 利用原基础镜像的工作台覆盖 |

### 恢复原版 WebUI

```bash
cd /data/minimax-h3/deploy
mv compose.override.yaml compose.override.yaml.disabled
docker compose up -d --no-build --force-recreate sglang-h3
```

本项目遵循 Apache License 2.0，许可文本见 `LICENSE`。原始上游：<https://github.com/sgl-project/sglang>，固定 SGLang 版本为 `v0.5.22`。


- 左上角品牌图标已替换为用户提供的透明吉祥物图。

### 等待动画说明

- **计时真实、进度不伪造**：只显示点击后经过的时间；滚动条和环形波纹是视觉动效，不表示 GPU 推理完成百分比。
- **不触碰业务**：按钮通过 `generate_with_history` 调用原 `generate(*args)`，仍使用 15 项既有输入和原视频输出；增加的回调仅记录元数据，不触发额外推理。
- **异常处理**：出现 Gradio 错误通知、生成按钮从运行状态恢复或新视频结果出现时，等待动画会退出。如果浏览器或特定 Gradio 版本无法检测任务结束，可使用“收起动画”手动关闭显示，不会停止模型。


### V4.2 定制修改

- 移除创作区的 “SYSTEM STATE / UI Motion Online” 装饰卡片，两张剩余卡片自动铺满宽度。
- 高级参数中的 **推理步数默认从 50 调整为 2**（原滑块最小值；最大 100、步长 1 不变）。更低步数可能降低画面质量，需要时可自行调高。
- 沿用 V4.1 的生成等待动画、日夜主题、吉祥物、15 项输入、参考图开关及视频结果展示。
- 此次仍不修改推理核心、请求构造或调度接口；V4.3 在 WebUI 回调外围增加 SQLite 记录，并扩展 Docker 覆盖安装。
