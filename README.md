# Multimodal Multitask Playground

This package implements commonly used blocks for multimodal and multitask learning. This is seperated out into encoders, decoders, fusion gates, task heads, and misc. The goal is to provide a modular and extensible framework.



We can abstract any multimodal model into the following components:

1. **Encoders**: each modality is encoded into a feature vector (embedding)
2. **Fusion** (optional): a method to fuse the feature vectors into a single (or multiple) representations
3. **Heads/Decoders**: uses fused representation(s) to produce task-specific outputs

In this package, we make assumptions about user-implemented modules. Each encoder's forward method accepts a tensor (the modality), and outputs a tensor (the embedding). The fusion module's forward method accepts a (consistently ordered) list of tensors (the embeddings), and outputs a tensor(s) (the fused representation(s)). The head's forward method accepts a tensor (the fused representation), and outputs a tensor (the task-specific output).

