# Idea: Video Prompt Echoing for 3D Spatial Reasoning

### Core Hypothesis
Causal Vision-Language Models (VLMs) suffer from the **"Question-First Paradox"**, where placing a textual prompt before an image successfully directs spatial visual encoding, but causes the final answer token to hallucinate due to long-context attention degradation across a massive block of visual tokens.

Because **3D video spatial reasoning** relies on incredibly long sequences (scaling from thousands to tens of thousands of spatiotemporal visual tokens per video clip), the text-to-space grounding bottleneck is severely amplified. Applying **Prompt Echoing** to video inputs can resolve this by keeping the spatial query fresh in the model's short-term memory (KV Cache) immediately before generation, without losing the question-driven visual focus during encoding.

---

### Implementation / Prompt Template

Instead of using standard single-pass ordering, structure the interleaved token sequence as a sandwich layout:

* [System Prompt]
* [3D Spatial Question]
* [Raw Video / Spatiotemporal Patch Tokens (10k+ tokens)]
* [3D Spatial Question Repeated (Echo)]
* [Assistant Answer Generation]

#### Example Framework Wrapper

```python
# Video Prompt Echoing Structure
messages = [
    {
        "role": "user",
        "content": [
            # 1. Initial Prompt: Directs the model's 3D/spatial visual focus during encoding
            {"type": "text", "text": "Locate the chair behind the desk. What are its relative 3D coordinates (x, y, z) to the camera at timestamp 0:12?"},
            
            # 2. Dense Video Stream: Emits thousands of visual tokens
            {"type": "video", "video": "path/to/3d_room_walkthrough.mp4"},
            
            # 3. Echoed Prompt: Places the context right next to the generation head to prevent recency bias
            {"type": "text", "text": "Reviewing the video above, what are the relative 3D coordinates (x, y, z) of the chair behind the desk at timestamp 0:12? Answer precisely:"}
        ]
    }
]
```

---

### Primary Theoretical Foundations

*   **Prompt Echoing Origin & Diagnostics:** 
    *Abhinandan et al., 2026.* ["Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models"](https://arxiv.org) (ECCV 2026 eXCV Workshop).
*   **Video MLLM Architecture Context:** 
    *   Utilises **Interleaved-MRoPE** (Multi-Resolutional Rotary Position Embedding) to align the 1D text positions directly with the 3D spatiotemporal video grid.