# 3D Perception Injection into VLMs (Flamingo-inspired Architecture)

## Core Idea
The goal is to enhance a Vision-Language Model's (VLM) awareness of 3D spaces from video input (using features like those from VGGT). 
Current state-of-the-art either uses shallow embedding projection, single-layer cross attention, or multi-layer modifications limited to early layers. 

**The Hypothesis:** Combining deep, multi-layer injection with cross-attention will outperform existing approaches. The architecture will draw inspiration from Flamingo (which fuses 2D into an LLM), but will be adapted to fuse 3D features into an already capable VLM.

## Architecture Specifics & Injection Strategy
*   **Frozen Backbone:** Keep the base VLM entirely frozen to prevent catastrophic forgetting of its 2D visual and linguistic priors.
*   **Gated Cross-Attention:** Interleave newly initialized gated cross-attention layers deep into the VLM (e.g., using `tanh(alpha)` initialized to zero for a smooth training ramp-up).
*   **Attention Formulation:** 
    *   **Queries (Q):** VLM's hidden states (2D patch tokens and text).
    *   **Keys (K) & Values (V):** 3D features extracted from the video.

## Key Design Decisions & Current Leans

### 1. Feature Compression: Proportional Downsampling over Perceiver Resampler
*   **Considered:** A fixed-size Perceiver Resampler (like in Flamingo) vs. proportional spatial downsampling.
*   **Leaning Towards:** Proportional downsampling (e.g., ViT-style patch merging, pooling, or strided convolutions).
*   **Rationale:** While a Perceiver aggressively reduces token count, it collapses features into an unstructured "soup," destroying the spatial grid. Proportional downsampling reduces the computational cost of cross-attention while preserving the strict spatial layout. This grid preservation is mandatory for applying accurate positional encodings in the next step.

### 2. Positional Encodings: 2D RoPE over Global 3D Coordinates
*   **Considered:** Global 3D positional encodings vs. 2D Rotary Positional Embeddings (RoPE) mapped to frame coordinates.
*   **Leaning Towards:** 2D RoPE based on frame coordinates.
*   **Rationale:** The underlying features (from VGGT) *already* encode intrinsic 3D geometry (depth, camera pose, object scale). The main hurdle is spatial alignment between these rich 3D features and the VLM's 2D image patches. By applying 2D RoPE to both the VLM's 2D patches (Queries) and the 3D features mapped to the 2D frame (Keys), it provides a strong inductive bias that perfectly aligns them (e.g., forcing the top-left 3D feature to map to the top-left 2D patch).