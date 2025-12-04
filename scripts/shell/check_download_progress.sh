#!/bin/bash
# Check download progress for the Hateful Memes images

IMG_DIR="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache/hateful_memes_images/img"

echo "=========================================="
echo "Hateful Memes download progress"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Inspect job status
echo "Job status:"
squeue -u zhisheng | grep hm_download || echo "  Download job finished or not running"
echo ""

# Summarize downloaded images
if [ -d "$IMG_DIR" ]; then
    COUNT=$(find "$IMG_DIR" -name "*.png" 2>/dev/null | wc -l)
    SIZE=$(du -sh "$IMG_DIR" 2>/dev/null | cut -f1)
    echo "Downloaded images:"
    echo "  Count: $COUNT files"
    echo "  Size: $SIZE"
    echo "  Path: $IMG_DIR"
else
    echo "Image directory not created yet"
fi

echo ""
echo "=========================================="

# If the download job is still running, show the tail of its logs
if squeue -u zhisheng | grep -q hm_download; then
    echo "Latest log output:"
    tail -20 /home/zhisheng/stats507/logs/jobs/hm_download_*.out 2>/dev/null | tail -10
fi

