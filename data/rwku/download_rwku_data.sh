#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pip install --upgrade --no-cache-dir gdown
gdown -c https://drive.google.com/uc?id=1ukWg-T3GPvqpyW7058vNyRWdXuQHRJPb

unzip RWKU.zip -d "$SCRIPT_DIR/benchmark"
rm RWKU.zip
