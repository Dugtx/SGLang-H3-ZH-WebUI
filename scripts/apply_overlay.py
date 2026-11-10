import hashlib
import importlib.util
import json
from pathlib import Path

manifest = json.loads(Path("/tmp/h3_zh_manifest.json").read_text())
spec = importlib.util.find_spec("sglang")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("SGLang package missing")
pkg = Path(next(iter(spec.submodule_search_locations)))
path = pkg / "multimodal_gen/apps/webui/minimax_h3.py"
if not path.is_file():
    raise SystemExit(f"H3 WebUI missing: {path}")
sha = hashlib.sha256(path.read_bytes()).hexdigest()
if sha != manifest["upstream_sha256"]:
    raise SystemExit(f"SGLang H3 source mismatch: {sha}")
localized = Path("/tmp/minimax_h3_zh.py")
if hashlib.sha256(localized.read_bytes()).hexdigest() != manifest["localized_sha256"]:
    raise SystemExit("Chinese overlay SHA256 mismatch")
path.write_bytes(localized.read_bytes())
print(f"Chinese UI installed: {path}")

# Install the isolated, stdlib-only history module after validating its checksum.
history_source = Path("/tmp/h3_studio_history.py")
if hashlib.sha256(history_source.read_bytes()).hexdigest() != manifest["history_sha256"]:
    raise SystemExit("H3 history module SHA256 mismatch")
history_destination = pkg / "multimodal_gen/apps/webui/h3_studio_history.py"
history_destination.write_bytes(history_source.read_bytes())
print(f"H3 Studio history installed: {history_destination}")
