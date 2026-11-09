#!/usr/bin/env bash
# Update the existing SGLang MiniMax-H3 Chinese WebUI without Git.
# Only the WebUI source and compose override are changed.
set -euo pipefail

DEPLOY_DIR="/data/minimax-h3/deploy"
UI_DIR="$DEPLOY_DIR/sglang-h3-zh-webui"
ARCHIVE_URL="https://codeload.github.com/Dugtx/SGLang-H3-ZH-WebUI/tar.gz/refs/heads/main"
RELEASE="v0.5.22-zh4"

for bin in curl tar python3 docker; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "缺少命令: $bin" >&2
    exit 1
  fi
done
if [ ! -f "$DEPLOY_DIR/compose.yaml" ]; then
  echo "找不到原来的 compose.yaml: $DEPLOY_DIR/compose.yaml" >&2
  exit 1
fi
if ! docker image inspect local/sglang-h3-native:v0.5.22 >/dev/null 2>&1; then
  echo "未找到基础镜像 local/sglang-h3-native:v0.5.22，现有服务未修改。" >&2
  exit 1
fi

STAGE="$(mktemp -d /tmp/h3-webui-update.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/source"

echo "[1/5] 下载 GitHub 最新源码（无需 Git）"
curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 \
  "$ARCHIVE_URL" -o "$STAGE/repo.tar.gz"
tar -xzf "$STAGE/repo.tar.gz" -C "$STAGE/source" --strip-components=1

echo "[2/5] 校验源码版本和完整性"
(
  cd "$STAGE/source"
  python3 scripts/build_overlay.py
  python3 tests/test_localization.py
  python3 -c 'import json; from pathlib import Path; m=json.loads(Path("manifest.json").read_text()); assert m["release"] == "v0.5.22-zh4", m["release"]; print("VERSION OK:", m["release"])'
)

echo "[3/5] 更新部署目录"
mkdir -p "$UI_DIR"
if [ -f "$DEPLOY_DIR/compose.override.yaml" ] && [ ! -f "$DEPLOY_DIR/compose.override.yaml.bak-before-zh4" ]; then
  /bin/cp -p "$DEPLOY_DIR/compose.override.yaml" "$DEPLOY_DIR/compose.override.yaml.bak-before-zh4"
fi
/bin/cp -a "$STAGE/source/." "$UI_DIR/"
/bin/cp -f "$UI_DIR/compose.override.yaml" "$DEPLOY_DIR/compose.override.yaml"

cd "$DEPLOY_DIR"
docker compose config --quiet

echo "[4/5] 构建 $RELEASE 镜像（推理服务此时保持运行）"
docker compose build sglang-h3

echo "[5/5] 重建容器并启动"
docker compose up -d --no-build --force-recreate sglang-h3

echo "部署命令已完成: $RELEASE"
docker ps --filter name=sglang-h3
echo "WebUI: http://<服务器地址>:18080"
