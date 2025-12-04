#!/bin/bash
#SBATCH --job-name=hm_git_lfs
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

mkdir -p /home/zhisheng/stats507/logs/jobs

CACHE_DIR="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
DATASET_DIR="${CACHE_DIR}/hateful_memes_full"

echo "=========================================="
echo "Download the full dataset via Git LFS"
echo "=========================================="
echo "Start time: $(date)"
echo "Destination: ${DATASET_DIR}"
echo ""

# Load git module
module load git

# Ensure git-lfs is available
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    # Install git-lfs under the user home directory
    cd /tmp
    wget -q https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz
    tar -xzf git-lfs-linux-amd64-v3.4.0.tar.gz
    mkdir -p ~/bin
    mv git-lfs-3.4.0/git-lfs ~/bin/
    export PATH=~/bin:$PATH
    git lfs install
    echo "✓ git-lfs installed"
else
    echo "✓ git-lfs already available"
    git lfs install
fi

echo ""

# Create cache directory
mkdir -p "${CACHE_DIR}"
cd "${CACHE_DIR}"

# Remove stale checkout if it exists
if [ -d "${DATASET_DIR}" ]; then
    echo "Removing previous dataset copy..."
    rm -rf "${DATASET_DIR}"
fi

echo "Cloning dataset repo (with LFS blobs)..."
echo "Expected transfer: roughly 3-4 GB."
echo ""

# Force git clone to download LFS blobs
GIT_LFS_SKIP_SMUDGE=0 git clone --progress https://huggingface.co/datasets/neuralcatcher/hateful_memes "${DATASET_DIR}"

echo ""
echo "=========================================="
echo "Download complete. Validating..."
echo "=========================================="

# Inspect the images directory
if [ -d "${DATASET_DIR}/img" ]; then
    IMG_COUNT=$(find "${DATASET_DIR}/img" -type f -name "*.png" 2>/dev/null | wc -l)
    IMG_SIZE=$(du -sh "${DATASET_DIR}/img" 2>/dev/null | cut -f1)
    echo "✓ Image count: ${IMG_COUNT}"
    echo "✓ Directory size: ${IMG_SIZE}"
    echo "✓ Location: ${DATASET_DIR}/img"
    
    # Spot-check a few files to ensure they are real images, not LFS pointers
    echo ""
    echo "Validating the first three image files:"
    count=0
    for img in "${DATASET_DIR}/img"/*.png; do
        if [ -f "$img" ]; then
            size=$(stat -c%s "$img" 2>/dev/null || stat -f%z "$img" 2>/dev/null)
            if [ $size -gt 1000 ]; then
                echo "  ✓ $(basename $img): ${size} bytes (binary image)"
            else
                echo "  ✗ $(basename $img): ${size} bytes (likely an LFS pointer)"
                head -5 "$img"
            fi
            count=$((count + 1))
            if [ $count -ge 3 ]; then
                break
            fi
        fi
    done
else
    echo "✗ Directory img/ not found"
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="


