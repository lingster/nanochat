#!/bin/bash
#
# Simple, proven HuggingFace CLI download script
#
# This script uses huggingface-cli for maximum reliability and speed.
# It downloads datasets to local storage, then converts to training format.
#
# Usage:
#   ./scripts/download_with_cli.sh codeparrot-clean 8
#   ./scripts/download_with_cli.sh the-stack 35
#   ./scripts/download_with_cli.sh github-code 15

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default output directory
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/archive/nanochat}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/mnt/archive/nanochat/downloads}"

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Check dependencies
check_dependencies() {
    print_msg "$YELLOW" "Checking dependencies..."

    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        print_msg "$YELLOW" "Installing huggingface-cli..."
        pip install -q huggingface_hub[cli]
    fi

    # Check if Python and required packages are available
    if ! python -c "import pyarrow, tqdm" 2>/dev/null; then
        print_msg "$YELLOW" "Installing Python dependencies..."
        pip install -q pyarrow tqdm
    fi

    print_msg "$GREEN" "✓ All dependencies installed"
}

# Download dataset using huggingface-cli
download_dataset() {
    local dataset_name=$1
    local dataset_path=$2
    local local_dir=$3
    local max_files=$4

    print_msg "$GREEN" "Downloading ${dataset_name}..."
    print_msg "$YELLOW" "This may take a while depending on dataset size and connection speed..."

    # Create download directory
    mkdir -p "$local_dir"

    # Download with huggingface-cli
    if [ -n "$max_files" ]; then
        huggingface-cli download "$dataset_path" \
            --repo-type dataset \
            --local-dir "$local_dir" \
            --max-workers 4 \
            --resume-download \
            --max-files "$max_files" || {
                print_msg "$RED" "Download failed. Check your internet connection and dataset name."
                exit 1
            }
    else
        huggingface-cli download "$dataset_path" \
            --repo-type dataset \
            --local-dir "$local_dir" \
            --max-workers 4 \
            --resume-download || {
                print_msg "$RED" "Download failed. Check your internet connection and dataset name."
                exit 1
            }
    fi

    print_msg "$GREEN" "✓ Download complete"
}

# Convert downloaded data to parquet format
convert_to_parquet() {
    local input_dir=$1
    local output_dir=$2
    local max_gb=$3

    print_msg "$GREEN" "Converting to parquet format..."

    # Create converter script path
    local converter="$(dirname "$0")/convert_to_parquet.py"

    if [ ! -f "$converter" ]; then
        print_msg "$RED" "Error: Converter script not found: $converter"
        print_msg "$YELLOW" "Please ensure convert_to_parquet.py exists in scripts/"
        exit 1
    fi

    python "$converter" \
        --input "$input_dir" \
        --output "$output_dir" \
        --max-gb "$max_gb" || {
            print_msg "$RED" "Conversion failed"
            exit 1
        }

    print_msg "$GREEN" "✓ Conversion complete"
}

# Get dataset info based on key
get_dataset_info() {
    local dataset_key=$1

    case "$dataset_key" in
        codeparrot-clean)
            echo "codeparrot/codeparrot-clean"
            ;;
        the-stack)
            echo "bigcode/the-stack-dedup"
            ;;
        github-code)
            echo "codeparrot/github-code"
            ;;
        *)
            print_msg "$RED" "Unknown dataset: $dataset_key"
            print_msg "$YELLOW" "Available datasets: codeparrot-clean, the-stack, github-code"
            exit 1
            ;;
    esac
}

# Main script
main() {
    if [ $# -lt 2 ]; then
        cat << EOF
Usage: $0 <dataset> <max_gb>

Downloads Python training data using HuggingFace CLI.

Arguments:
    dataset     Dataset to download (codeparrot-clean, the-stack, github-code)
    max_gb      Maximum size to download in GB

Environment variables:
    OUTPUT_DIR      Output directory for parquet files (default: /mnt/archive/nanochat)
    DOWNLOAD_DIR    Temporary download directory (default: /mnt/archive/nanochat/downloads)

Examples:
    # Download 8GB of codeparrot-clean
    $0 codeparrot-clean 8

    # Download 35GB of The Stack to custom directory
    OUTPUT_DIR=/custom/path $0 the-stack 35

    # Download 15GB of GitHub Code
    $0 github-code 15

Available datasets:
    codeparrot-clean  : Clean Python code from GitHub (~8GB recommended)
    the-stack         : The Stack Dedup Python subset (~35GB recommended)
    github-code       : GitHub Code Python (~15GB recommended)

Notes:
    - Downloads can be resumed if interrupted
    - Uses multiple workers for faster downloads
    - Automatically converts to parquet format
    - Progress is shown during download and conversion
EOF
        exit 1
    fi

    local dataset_key=$1
    local max_gb=$2
    local dataset_path=$(get_dataset_info "$dataset_key")
    local download_dir="$DOWNLOAD_DIR/$dataset_key"

    print_msg "$GREEN" "================================"
    print_msg "$GREEN" "HuggingFace CLI Download Script"
    print_msg "$GREEN" "================================"
    echo ""
    print_msg "$YELLOW" "Dataset:      $dataset_path"
    print_msg "$YELLOW" "Max size:     ${max_gb} GB"
    print_msg "$YELLOW" "Download dir: $download_dir"
    print_msg "$YELLOW" "Output dir:   $OUTPUT_DIR"
    echo ""

    # Check dependencies
    check_dependencies
    echo ""

    # Download dataset
    download_dataset "$dataset_key" "$dataset_path" "$download_dir" ""
    echo ""

    # Convert to parquet
    convert_to_parquet "$download_dir" "$OUTPUT_DIR" "$max_gb"
    echo ""

    # Show summary
    print_msg "$GREEN" "================================"
    print_msg "$GREEN" "Download Complete!"
    print_msg "$GREEN" "================================"
    echo ""
    print_msg "$YELLOW" "Output location: $OUTPUT_DIR"
    print_msg "$YELLOW" "Downloaded:      $(du -sh "$download_dir" 2>/dev/null | cut -f1)"
    print_msg "$YELLOW" "Converted:       $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"
    echo ""
    print_msg "$GREEN" "You can now use these files for training!"
    print_msg "$YELLOW" "Tip: You can delete $download_dir to save space"
    echo ""
}

main "$@"
