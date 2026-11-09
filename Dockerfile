FROM local/sglang-h3-native:v0.5.22

COPY overlay/minimax_h3.py /tmp/minimax_h3_zh.py
COPY manifest.json /tmp/h3_zh_manifest.json
COPY scripts/apply_overlay.py /tmp/apply_overlay.py

RUN python /tmp/apply_overlay.py

LABEL org.opencontainers.image.title="SGLang MiniMax-H3 Chinese WebUI" \
      org.opencontainers.image.version="v0.5.22-zh4"
