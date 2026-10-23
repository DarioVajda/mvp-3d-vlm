#!/bin/bash
# Download and extract the media for SpatialStack's training mix (third_party/SpatialStack/TRAINING.md); run inside a
# CPU job. The annotations are in data/raw/SpatialStackData/. Archives go to data/raw/<repo>/, the extracted media to
# data/train_media/, laid out as the annotations' paths expect (spar/, llava_hound/frames/, scannet/videos/).
# Safe to rerun: downloads resume and finished extractions are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=.venv/bin/hf
MEDIA=data/train_media
mkdir -p $MEDIA/llava_hound/frames $MEDIA/scannet
command -v pigz >/dev/null && GZ=pigz || GZ=gzip   # the container has no pigz

# ScanNet training videos (VLM-3R QA and VSI-590K appearance order; the same 1201 scenes as Spatial-MLLM's 133k)
$HF download Journey9ni/aweb --repo-type dataset --include "ScanNet/videos/train/**" --local-dir data/raw/aweb &
# SPAR-7M images, at the revision TRAINING.md pins; one tar.gz split into 16 chunks
$HF download jasonzhango/SPAR-7M --repo-type dataset --revision 976c19177468eabe64e9e2dd0f0450cd32dacc1f \
    --include "spar-*.tar.gz" --local-dir data/raw/SPAR-7M &
# LLaVA-Hound frames (300k videos, of which llava_hound_64k uses 64k)
$HF download ShareGPTVideo/train_video_and_instruction --repo-type dataset --include "train_300k/**" \
    --local-dir data/raw/LLaVA-Hound &
wait
echo "[$(date -u +%FT%TZ)] downloads done"; du -sh data/raw/aweb data/raw/SPAR-7M data/raw/LLaVA-Hound

ln -sfn ../../raw/aweb/ScanNet/videos/train $MEDIA/scannet/videos
echo "scannet videos: $(ls $MEDIA/scannet/videos/ | wc -l)"

if [ ! -e $MEDIA/.spar_done ]; then
    ls data/raw/SPAR-7M/spar-*.tar.gz | sort -V | xargs cat | $GZ -dc | tar -xf - -C $MEDIA
    [ -d $MEDIA/spar/spar ] && { mv $MEDIA/spar/spar/* $MEDIA/spar/ && rmdir $MEDIA/spar/spar; }
    touch $MEDIA/.spar_done
fi
echo "[$(date -u +%FT%TZ)] spar extracted: $(ls $MEDIA/spar)"

if [ ! -e $MEDIA/.llava_hound_done ]; then
    find data/raw/LLaVA-Hound/train_300k -maxdepth 1 -name 'chunk_*.tar.gz' -print0 \
        | xargs -0 -P"$(nproc)" -I{} tar -I $GZ -x -f {} -C $MEDIA/llava_hound/frames
    touch $MEDIA/.llava_hound_done
fi
echo "[$(date -u +%FT%TZ)] llava_hound extracted: $(ls $MEDIA/llava_hound/frames | wc -l) entries"
