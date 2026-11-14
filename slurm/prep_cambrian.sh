#!/usr/bin/env bash
# Unpack the downloaded Cambrian-S video archives (data/raw/Cambrian-S-3M/<source>_NNN.tar.zst) into
# data/train_media/cambrian/<source>/, as the dataset's decompress.sh lays them out, so that a sample's media is
# cambrian/<its "video" field>. Skips archives already unpacked (a marker per archive). CPU-only job:
#   LOGDIR=prep slurm/submit.sh -A lkm -J prep_cambrian -t 06:00:00 --gres gpu:0 -c 16 --mem 32G --constraint '' \
#       -- bash slurm/prep_cambrian.sh
set -euo pipefail
src=data/raw/Cambrian-S-3M
dst=data/train_media/cambrian
mkdir -p "$dst/.done"
unpack() {
    local f=$1 b d
    b=$(basename "$f" .tar.zst); d=$dst/${b%_[0-9][0-9][0-9]}
    [[ -f $dst/.done/$b ]] && return 0
    mkdir -p "$d"
    tar -I zstd --no-same-owner -xf "$f" -C "$d" && touch "$dst/.done/$b" && echo "$b -> $d"
}
export -f unpack; export dst
ls "$src"/*.tar.zst | xargs -P 8 -I{} bash -c 'unpack {}'
echo "unpacked $(ls "$dst/.done" | wc -l) of $(ls "$src"/*.tar.zst | wc -l) archives"
