#!/usr/bin/env python3
"""
網羅的なデータリーケージチェック
各特徴量がviewsを予測する上で適切かを確認
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("網羅的データリーケージチェック")
print("="*60)

# データ読み込み
df = pd.read_csv('youtube_top_new_complete.csv')

# 全ての特徴量をリストアップ
all_features = [
    'video_duration',
    'tags_count', 
    'description_length',
    'subscribers',
    'object_complexity',
    'element_complexity',
    'brightness',
    'colorfulness',
    'time_duration',
    'published_at',
    'keyword',
    'thumbnail_path'
]

print("【各特徴量の評価】\n")

# 1. video_duration
print("1. video_duration（動画の長さ）")
print("   - アップロード時に決まる ✓")
print("   - viewsより前に存在 ✓") 
print("   - 結論: 安全 ✓\n")

# 2. tags_count
print("2. tags_count（タグ数）")
print("   - アップロード時に設定 ✓")
print("   - viewsより前に存在 ✓")
print("   - 結論: 安全 ✓\n")

# 3. description_length
print("3. description_length（説明文の長さ）")
print("   - アップロード時に設定 ✓")
print("   - viewsより前に存在 ✓")
print("   - 結論: 安全 ✓\n")

# 4. subscribers - これが問題かも？
print("4. subscribers（チャンネル登録者数）")
print("   - 現在の登録者数？過去の登録者数？")
# 実際のデータで確認
print(f"   - subscribersの範囲: {df['subscribers'].min():,} ~ {df['subscribers'].max():,}")
print(f"   - viewsの範囲: {df['views'].min():,} ~ {df['views'].max():,}")
# 相関をチェック
corr = df['subscribers'].corr(df['views'])
print(f"   - subscribersとviewsの相関: {corr:.4f}")
print("   - 問題: 動画が人気→登録者増加の可能性")
print("   - 結論: 要注意 ⚠️\n")

# 5. log_subscribers
print("5. log_subscribers（log変換した登録者数）")
print("   - subscribersの対数変換")
print("   - subscribersと同じ問題を持つ")
print("   - 結論: 要注意 ⚠️\n")

# 6. サムネイル関連
print("6. brightness, colorfulness, object_complexity, element_complexity")
print("   - サムネイル画像から抽出 ✓")
print("   - アップロード時に決まる ✓")
print("   - 結論: 安全 ✓\n")

# 7. published_at / days_since_publish
print("7. published_at / days_since_publish")
print("   - 投稿日時 ✓")
print("   - viewsより前に決まる ✓")
print("   - 結論: 安全 ✓\n")

# 8. time_duration
print("8. time_duration")
df['time_duration_int'] = pd.to_numeric(df['time_duration'], errors='coerce')
print(f"   - データ型: {df['time_duration'].dtype}")
print(f"   - サンプル: {df['time_duration'].head().tolist()}")
print("   - 内容が不明...")
print("   - 結論: 要確認 ?\n")

# 派生特徴量のチェック
print("【派生特徴量の評価】\n")

print("9. subscriber_per_view")
print("   - subscribers / views で計算")
print("   - viewsを使って計算している！")
print("   - 結論: データリーケージ ✗\n")

print("10. tags_per_second, desc_per_second")
print("   - tags_count / video_duration など")
print("   - 両方ともアップロード時の情報 ✓")
print("   - 結論: 安全 ✓\n")

# subscribersの時系列問題を詳しく調査
print("\n【subscribersの時系列問題の詳細調査】")
# published_atをパース
df['published_at_parsed'] = pd.to_datetime(df['published_at'])
df['year'] = df['published_at_parsed'].dt.year
df['month'] = df['published_at_parsed'].dt.month

# 年ごとのsubscribers分布
print("\n年ごとの平均subscribers:")
yearly_stats = df.groupby('year')['subscribers'].agg(['mean', 'median', 'count'])
print(yearly_stats)

print("\n【最終判定】")
print("安全な特徴量:")
safe_features = [
    'video_duration', 'tags_count', 'description_length',
    'object_complexity', 'element_complexity', 'brightness', 'colorfulness',
    'days_since_publish', 'tags_per_second', 'desc_per_second'
]
for f in safe_features:
    print(f"  ✓ {f}")

print("\n要注意/除外すべき特徴量:")
risky_features = [
    'subscribers（現在の値の可能性）',
    'log_subscribers（同上）',
    'subscriber_per_view（明確なリーケージ）',
    'time_duration（内容不明）'
]
for f in risky_features:
    print(f"  ⚠️ {f}")

print("\n推奨: subscribersを使わないモデルを作成すべき")