#!/bin/bash
# Sync source files to addon directory for building

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_DIR="$SCRIPT_DIR/speaker_recognition_addon"

echo "Syncing source files to addon directory..."

# Remove old files
rm -rf "$ADDON_DIR/pyproject.toml" "$ADDON_DIR/speaker_recognition"

# Copy current files
cp "$SCRIPT_DIR/pyproject.toml" "$ADDON_DIR/"
cp -r "$SCRIPT_DIR/speaker_recognition" "$ADDON_DIR/"

echo "✓ Source files synced successfully"
