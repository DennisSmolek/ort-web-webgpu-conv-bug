# Draft for the microsoft/onnxruntime issue — copy/paste below the line

Suggested labels: `ep:WebGPU`, `platform:web`
File at: https://github.com/microsoft/onnxruntime/issues/new (Bug Report / Web)

---

**Title:** `[Web/WebGPU EP] First Conv reducing a >3-channel graph input miscomputes (error scales with channel count; WASM/CPU correct)`

### Describe the issue

In onnxruntime-web **1.27.0**, the WebGPU execution provider produces wrong
results for the **first `Conv` that reduces the model's raw graph-input tensor
when that input has more than 3 channels**. The WASM provider, the Python CPU
provider, and the model's native reference implementation all agree on the same
model + input; only the WebGPU EP diverges.

Observed on the Intel Open Image Denoise U-Nets exported to ONNX (opset 17) —
identical topology, differing only in graph-input channel count:

| model input channels | max\|Δ\| WebGPU vs WASM | verdict |
|---|---|---|
| 3 | 9.5e-7 | ✅ matches |
| 6 | 3.7e-2 | ❌ diverges |
| 9 | 1.1e-1 | ❌ diverges |
| 9 (fp16 export) | 1.1e-1 | ❌ diverges — fp16 does not dodge it |

Key isolation result: it is **specifically the graph-input Conv**, not wide
convs in general. The same network reduces the same raw 9-channel input a
second time mid-graph (`dec_conv1a`, fed by `Concat(up, input)` = 105
channels). Splitting the graph after the first conv and running **everything
else — including that 105-channel raw-input conv — on WebGPU** matches the
WASM reference to **1.19e-6**. So all internal convs (32–256ch) and even wider
raw-input-consuming convs are fine; only the first conv touching the raw >3ch
graph input miscomputes. This points at how the EP prepares/packs the graph
input for the initial Conv (layout / channel tiling / input-channel
reduction), not at the Conv kernel in general.

Conditions ruled out: graph capture off, plain CPU-array IO (no gpu-buffer
tensors, no freeDimensionOverrides), fp32 and fp16, two model sizes. The ONNX
files themselves match Intel's native OIDN output to ~54 dB on the CPU/WASM
providers.

### To reproduce

Standalone repro repo (two click-and-run browser tests, no build step):
**https://github.com/DennisSmolek/ort-web-webgpu-conv-bug**

- [`repro/`](https://github.com/DennisSmolek/ort-web-webgpu-conv-bug/tree/main/repro) —
  bare vanilla ORT-web sessions, one fixed synthetic input, WebGPU vs WASM
  across 3/6/9-channel models (ORT + models load from public CDNs).
  `python3 -m http.server` and open — results print and land on `window.__results`.
- [`split/`](https://github.com/DennisSmolek/ort-web-webgpu-conv-bug/tree/main/split) —
  the isolation: same comparison with the graph split at the first conv,
  proving the rest of the network (including the wider raw-input conv) is
  correct on WebGPU.

### Urgency

Blocks aux-guided (albedo+normal, 9-channel) denoising on WebGPU for the
`denoiser` library (https://github.com/pmndrs/denoiser). We ship a verified
workaround (compute the first conv outside the EP, run a re-exported tail on
WebGPU — restores 1.2e-6 accuracy), but it costs a custom kernel + re-exported
models for every variant.

### Platform

- onnxruntime-web 1.27.0 (jsDelivr npm CDN build)
- Chrome (headed + headless), ANGLE/Metal, Apple M-series (macOS)
- Execution provider: WebGPU (JSEP)

### Possibly related

#24070, #26734, #24442
