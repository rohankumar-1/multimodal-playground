# Multimodal Multitask Playground

This package implements commonly used blocks for multimodal and multitask learning. It is separated into encoders, decoders, fusion gates, task heads, and misc. The goal is to provide a modular and extensible framework.

Install (library only):

```bash
pip install -e .
```

Install with dev tools (pytest, ruff):

```bash
pip install -e ".[dev]"
```

Run tests from the repository root:

```bash
pytest
```

If imports fail, ensure the package is installed as above or run `PYTHONPATH=src pytest`.



We can abstract any multimodal model into the following components:

1. **Encoders**: each modality is encoded into a feature vector (embedding)
2. **Fusion** (optional): a method to fuse the feature vectors into a single (or multiple) representations
3. **Heads/Decoders**: uses fused representation(s) to produce task-specific outputs

In this package, we make assumptions about user-implemented modules. Each encoder's forward method accepts a tensor (the modality), and outputs a tensor (the embedding). The fusion module's forward method accepts a (consistently ordered) list of tensors (the embeddings), and outputs a tensor(s) (the fused representation(s)). The head's forward method accepts a tensor (the fused representation), and outputs a tensor (the task-specific output).

Encoders accept (B, input_dim...) and output (B, latent_dim). The fusion layer accepts a list of tensors of shapes [(B, input_dim_i), (B, input_dim_j), ...] and outputs a tensor of shape (B, fusion_dim). The head's forward method accepts a tensor of shape (B, fusion_dim) and outputs a tensor of shape (B, output_dim).