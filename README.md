# onnxruntime-web WebGPU EP: first Conv on a >3-channel graph input miscomputes

**onnxruntime-web 1.27.0 — the WebGPU execution provider produces wrong results
for the first `Conv` that reduces the model's raw input tensor when that input
has more than 3 channels.** The WASM (CPU) provider, the Python CPU provider, and
Intel's native reference are all correct on the same model + input; only the
WebGPU EP diverges. The error scales with input-channel count and affects fp32
and fp16.

This repository is a minimal, self-contained reproduction to accompany an
onnxruntime issue. Two standalone browser tests, no build step:

- [`repro/`](repro/) — **the bug.** Bare vanilla ORT-web sessions (no wrapper
  code, plain CPU-array IO), one fixed synthetic input, WebGPU vs WASM provider,
  across 3/6/9-channel models. Models + ORT load from public CDNs — nothing local.
- [`split/`](split/) — **the isolation + workaround.** Narrows the fault to the
  *graph-input* Conv specifically (not "any wide conv"), and proves that moving
  just that one op off the WebGPU EP restores exact results.

## Summary of findings

The test models are the [Intel Open Image Denoise](https://github.com/RenderKit/oidn)
U-Nets converted to ONNX (opset 17). Every variant is the same topology; they
differ only in how many channels the **raw graph input** has (3 = color, 6 =
+albedo, 9 = +albedo+normal).

| model | input channels | max\|Δ\| WebGPU vs WASM | verdict |
|---|---|---|---|
| `rt_hdr` | **3** | 9.5e-7 | ✅ matches |
| `rt_hdr_alb` | **6** | 3.7e-2 | ❌ diverges |
| `rt_hdr_calb_cnrm` | **9** | 1.1e-1 | ❌ diverges |
| `rt_hdr_calb_cnrm_small` | **9** | 1.5e-1 | ❌ diverges |
| `rt_hdr_calb_cnrm.fp16` | **9** | 1.1e-1 | ❌ diverges (fp16 too) |

Divergence appears **above 3 input channels and scales with the count**. Every
model's *internal* convs have 32–256 channels and are fine (the 3-channel model
exercises all of them correctly), so it is not "wide convs" in general.

The `split/` test narrows it further. The U-Net reduces the raw 9-channel input
in **two** convs — `enc_conv0` (first layer, `9→32`) and `dec_conv1a` (a decoder
conv fed `concat(…, input)` = 105 channels including the raw input). Running the
**entire network except `enc_conv0`** on WebGPU (with a correct `enc_conv0`
supplied) gives **1.19e-6 vs the WASM reference** — exact. So `dec_conv1a`, a
*wider* conv that also ingests the raw input, is **correct** on WebGPU. The fault
is specifically the **model's first Conv reducing the raw graph-input tensor when
that input has >3 channels** — pointing at how the WebGPU EP prepares/packs the
graph input for that initial op (layout / channel tiling / the input-channel
reduction), not the Conv kernel in general.

## Environment

- onnxruntime-web **1.27.0** (loaded from jsDelivr npm CDN in the tests)
- Chrome (headless and headed), ANGLE/Metal backend, Apple M-series
- Reproduced fp32 + fp16, base + small model sizes
- Graph capture **off**; plain CPU-array IO (no gpu-buffer tensors, no
  `freeDimensionOverrides`) — the EP is the only variable

## Workaround (verified, in `split/`)

Compute the offending first conv (`enc_conv0`: `Conv 9→32` + `relu6`) **off the
WebGPU EP** — a small WGSL kernel, or on WASM — then run a re-exported tail that
starts at the second conv on WebGPU (two inputs: the first conv's feature map
**and** the raw `input`, which the late `dec_conv1a` skip still needs). Measured
end-to-end: **1.2e-6 vs the native/WASM reference** — full quality, no CPU
fallback for the bulk of the network. fp16 does **not** dodge the bug.

## Related

Possibly-related open reports: microsoft/onnxruntime #24070, #26734, #24442.

## License

Reproduction code: MIT. The `.onnx` models under `split/models/` are derived from
Intel Open Image Denoise weights (© Intel, **Apache-2.0**) — see [NOTICE](NOTICE).
