#!/usr/bin/env bash
# Second batch of 75 shared scenes: the next 75 by export.py's ranking (same quota per dataset), leaving out the
# first batch in out/. Exports, checks, adds the depth maps, zips both parts and uploads them next to the first batch
# on Google Drive (rclone remote gdrive:, folder mvp_vggt_share/). CPU only; runs inside the container from mvp/:
#   LOGDIR=vggt_data slurm/submit.sh -J vggt_batch2 -t 03:00:00 --gres gpu:0 -c 8 --mem 32G -- bash tasks/vggt_data/export_batch2.sh
set -euo pipefail
D=tasks/vggt_data
# scenes.csv is written last, so it marks a finished export (a resubmission goes on from the checks)
[[ -f $D/out_batch2/scenes.csv ]] || python $D/export.py --exclude $D/out/scenes.csv --out $D/out_batch2
cp $D/load_example.py $D/out_batch2/
cp $D/README_batch2.md $D/out_batch2/README.md
cp $D/out/README.md $D/out_batch2/FORMAT.md
python $D/check.py $D/out_batch2
python $D/export_depth.py --main $D/out_batch2 --out $D/out_depth_batch2
cp $D/README_depth.md $D/out_depth_batch2/README.md
cd $D
rm -f vggt_vsibench_batch2.zip depth_vsibench_batch2.zip
zipdir() {  # <source dir> <name in the zip> [store]: zips <name>/ (store: no compression, for the .npz depth files)
    ln -sfn "$1" "$2"
    if command -v zip > /dev/null; then zip -qr${3:+0} "$2.zip" "$2/"; else python -m zipfile -c "$2.zip" "$2/"; fi
    rm "$2"
}
zipdir out_batch2 vggt_vsibench_batch2
zipdir out_depth_batch2 depth_vsibench_batch2 store
ls -l *_batch2.zip
sha256sum *_batch2.zip
RCLONE=$HOME/bin/rclone
$RCLONE copy vggt_vsibench_batch2.zip gdrive:mvp_vggt_share/ --drive-chunk-size 128M
$RCLONE copy depth_vsibench_batch2.zip gdrive:mvp_vggt_share/ --drive-chunk-size 128M
$RCLONE lsl gdrive:mvp_vggt_share/
for z in vggt_vsibench_batch2.zip depth_vsibench_batch2.zip; do echo "$z $($RCLONE link gdrive:mvp_vggt_share/$z)"; done
