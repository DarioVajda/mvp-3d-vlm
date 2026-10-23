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
## Implementation Notes
*   **Data (first stage):** VLM-3R ScanNet QA only: 51.8k VSI-Bench-template questions on 1,201 ScanNet videos, none of them VSI-Bench test scenes. Half of it is relative direction, and there is no route planning. Appearance order (same videos) and the full SpatialStack mix come later; consider dropping SPAR's `_video_hard` samples, which name objects without any markers.
*   **Prompts** have no `<video>` placeholder ("These are frames of a video.\n..."), so the loader puts the visual tokens in front of the text, as SpatialStack does.
*   **Frames:** 32 uniform frames per video, and the same frames go to Qwen and VGGT. VGGT processes a set of frames jointly (global attention across frames, first frame as reference), not the whole video, and its outputs are scale-normalised, so metric answers have to be learned.
*   **VGGT features:** computed on the fly (frozen, `no_grad`, bf16; ~0.6 s and ~8 GB of GPU memory for 32 frames on an H100). No cache: one layer at full resolution is ~165 GB for the 1,201 videos, and the layer choice is still open. Per frame: 37×28 patch tokens at 518×392, 2048-d (frame and global attention outputs concatenated), plus 5 camera/register tokens.
*   **Frame alignment:** Qwen's video mode merges pairs of frames (temporal patch size 2; check for Qwen3.5), so each Qwen token attends to the VGGT features of its two frames.
*   **Controls:** zero-shot Qwen3.5-4B, plus the same trainable parameters, data, frames and steps without VGGT (e.g. cross-attending to Qwen's own vision features), so that gains from VGGT are separated from gains from fitting VSI's templates. Report per task, on the debiased split and on MMSI.
