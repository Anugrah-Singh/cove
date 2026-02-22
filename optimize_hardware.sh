#!/bin/bash
echo "🚀 VISION ARCHIVE: Flexible Hardware Optimization"

# Try to install hardware-specific faiss if possible
echo "Attempting to install faiss-gpu (if available for this Python version)..."
pip install faiss-gpu || echo "⚠️  faiss-gpu not available (likely due to Python version), falling back to faiss-cpu"

# Ensure at least faiss-cpu is installed
pip install faiss-cpu

# Ensure onnxruntime-gpu is installed (which usually works)
pip install onnxruntime-gpu

echo "✅ Optimization Complete."
echo "   The app will now automatically use GPU if available,"
echo "   and transparently fall back to high-performance CPU mode otherwise."
