# SGLang Ascend Quantization

This context defines the shared language for SGLang LLM and Diffusion quantization on Ascend NPU.

## Language

**Qwen MoE**:
The shared Mixture-of-Experts quantization target covering both Qwen3 and Qwen3.5 model families.
_Avoid_: Qwen3.5-specific MoE quantization

**W4A8 MXFP**:
An LLM quantization format pairing MXFP4 weights with dynamically quantized MXFP8 activations.
_Avoid_: W4A4, INT4 W4A8

**Online Quantization**:
A mode that starts from BF16 or FP16 weights and quantizes them during model initialization.
_Avoid_: Runtime checkpoint quantization

**Offline ModelSlim Quantization**:
A mode that starts from a pre-quantized ModelSlim checkpoint whose description selects the quantization scheme.
_Avoid_: Online ModelSlim, explicit ModelSlim scheme selection

## Relationships

- **Qwen MoE** uses the same quantization capabilities for Qwen3 and Qwen3.5
- **W4A8 MXFP** supports both **Online Quantization** and **Offline ModelSlim Quantization**
- **Online Quantization** and **Offline ModelSlim Quantization** share the same inference semantics

## Example dialogue

> **Dev:** "Do we need a separate Qwen3.5 MoE W4A8 implementation?"
> **Domain expert:** "No. Add **W4A8 MXFP** to the shared **Qwen MoE** capability, then validate both weight-source modes."

## Flagged ambiguities

- "Qwen3.5 MoE W4A8 adaptation" could imply a model-specific quantization path; resolved: it extends the shared **Qwen MoE** capability.
