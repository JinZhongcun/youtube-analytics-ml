#!/bin/bash
cd /mnt/c/Users/jinsm/kokusaiyugou

# PCA分析の実行
docker run --rm -v $(pwd):/work -w /work python:3.9 bash -c "
pip install -q pandas numpy matplotlib seaborn scikit-learn
python youtube_analysis.py
"

# SVM分析の実行
docker run --rm -v $(pwd):/work -w /work python:3.9 bash -c "
pip install -q pandas numpy matplotlib seaborn scikit-learn
python svm_analysis.py
"

echo "分析が完了しました。"
ls -la *.png *.json