"""Genera l'array C++ dal modello TFLite senza alterarne un solo byte."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
model = root / "model" / "dense_f32.tflite"
out_dir = Path(__file__).resolve().parent / "src"
data = model.read_bytes()

(out_dir / "model_data.h").write_text(
    "#pragma once\n#include <cstddef>\nextern const unsigned char g_model_data[];\n"
    "extern const size_t g_model_data_len;\n",
    encoding="ascii",
)
lines = ["#include \"model_data.h\"", "", "alignas(16) const unsigned char g_model_data[] = {"]
for offset in range(0, len(data), 12):
    lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in data[offset:offset + 12]) + ",")
lines.extend(["};", f"const size_t g_model_data_len = {len(data)};", ""])
(out_dir / "model_data.cc").write_text("\n".join(lines), encoding="ascii")
print(f"Array modello generato: {len(data)} byte")
