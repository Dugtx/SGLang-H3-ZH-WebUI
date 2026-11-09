# SGLang MiniMax-H3 简体中文 WebUI

针对 SGLang v0.5.22 原生 MiniMax-H3 Gradio 前端的简体中文本地化。保留原有推理路径，仅调整页面控件与图片选择行为，保留英文标题 SGLang MiniMax H3。用户要求移除的说明段落已完全删除。

- 中文输入标签、参数标签和生成按钮。
- 顶部模型路径和任务选择框隐藏，任务由后端分区决定，通过 Gradio State 保留事件参数。
- 不改模型权重、推理后端、API、CUDA 内核和原生 MiniMaxH3Pipeline，不依赖 ComfyUI 服务。
- Ref2VA 增加“是否选择上传图片”开关：默认关闭，自动附带启动时生成的透明 RGBA PNG；打开后必须上传参考图。
- 只支持 SGLang v0.5.22。构建时 SHA-256 校验原版和译文，源文件版本不符即中止。

## 在现有 openEuler 上更新（无需 Git）

在没有生成任务时，以 root 执行：

```bash
curl -fL --retry 3 --connect-timeout 15 \
  https://raw.githubusercontent.com/Dugtx/SGLang-H3-ZH-WebUI/main/scripts/deploy_no_git.sh \
  -o /tmp/h3-deploy-no-git.sh
bash /tmp/h3-deploy-no-git.sh
```

脚本下载源码压缩包、校验 v0.5.22-zh4、重建汉化覆盖文件并运行离线测试；**只有上述检查全部通过才更新部署目录**。之后更新 `compose.override.yaml`，构建新镜像并重新创建 `sglang-h3` 容器。

本仓库只覆盖 WebUI 文件，不修改 `compose.yaml`、模型权重、GPU 参数、端口或 Docker daemon。构建依赖服务器现有基础镜像 `local/sglang-h3-native:v0.5.22`。重建容器会中断正在运行的生成任务，须在空闲时操作。

### 恢复原版 WebUI

    cd /data/minimax-h3/deploy
    mv compose.override.yaml compose.override.yaml.disabled
    docker compose up -d --no-build --force-recreate sglang-h3

回退只切换镜像，不删除模型、权重或缓存。

## 参考图片开关（v0.5.22-zh4）

在 Ref2VA 页面上，开关默认关闭，参考图片上传组件隐藏；提交时自动附带一张启动时在容器临时目录生成的透明 PNG。开启后显示上传组件，未选择图片时拒绝提交。开关变化会清空旧上传内容，关闭时生成回调也会忽略旧图。关键帧、视频、音频、提示词及推理设置保持不变。该占位图仍属于 Ref2VA 条件，不等价于真正的 T2VA 模式。

## 仓库文件

- overlay/minimax_h3.py：中文原生 Gradio UI
- upstream/minimax_h3.py：相同标签的上游原始代码
- patches/minimax_h3_zh.patch：全部差异
- manifest.json：上游及译文 SHA-256
- scripts/build_overlay.py：重新生成译文及 patch
- scripts/deploy_no_git.sh：无需 Git 的下载、校验、构建和部署脚本
- scripts/apply_overlay.py：Docker 构建时校验并覆盖官方 UI
- tests/test_localization.py：语法、哈希、回调与关键控件测试
- Dockerfile：以本地已验证 SGLang 镜像为基础，覆盖单个 UI 文件
- compose.override.yaml：不侵入原有 Compose 服务的镜像切换

本地测试：

    python3 scripts/build_overlay.py
    python3 tests/test_localization.py

## 授权与来源

上游项目：https://github.com/sgl-project/sglang
固定版本：v0.5.22
修改的上游文件：python/sglang/multimodal_gen/apps/webui/minimax_h3.py

遵循 Apache License 2.0，LICENSE 包含原版许可全文。上游代码版权归 SGLang 贡献者。改动可通过 patch 审查。
