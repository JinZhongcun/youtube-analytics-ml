#!/usr/bin/env python3
"""
批判的査読（シンプル版）: 最も重要な問題に焦点
"""

import pandas as pd
import numpy as np

print("="*80)
print("批判的査読: 最重要問題の検証")
print("="*80)

# データ読み込み
df_old = pd.read_csv('youtube_top_new.csv')
df_new = pd.read_csv('youtube_top_new_complete.csv')

print("\n【問題1: データの不整合】")
print(f"元データ: {len(df_old)}件")
print(f"新データ: {len(df_new)}件")
print(f"共通video_id: {len(set(df_old['video_id']) & set(df_new['video_id']))}件")
print(f"失われたデータ: {len(set(df_old['video_id']) - set(df_new['video_id']))}件")

# マージして検証
merged = pd.merge(df_old, df_new, on='video_id', suffixes=('_old', '_new'))

print("\n【問題2: viewsの変化】")
views_changed = (merged['views_old'] != merged['views_new']).sum()
print(f"viewsが変化した動画: {views_changed}/{len(merged)} ({views_changed/len(merged)*100:.1f}%)")

# 変化の大きさ
merged['views_change_pct'] = (merged['views_new'] - merged['views_old']) / merged['views_old'] * 100
print(f"平均変化率: {merged['views_change_pct'].mean():.2f}%")

print("\n【問題3: データ収集時期の不一致】")
print("これは同じ動画の異なる時点のデータです！")
print("→ subscribersは「現在の値」であり、動画公開時の値ではない")
print("→ これがモデルの予測に影響している可能性大")

print("\n【批判的結論】")
print("1. 元データと新データは異なる時期に収集されている")
print("2. subscribersは動画公開時ではなく、データ収集時の値")
print("3. これは時系列リーケージの一種")
print("4. 正しい分析には、動画公開時のsubscribers数が必要")

# Geminiに相談する内容を準備
print("\n【Geminiへの相談内容】")
consult_text = """
# YouTube動画分析の時系列リーケージ問題

## 状況
- ターゲット: views（再生数）
- 特徴量: subscribers（チャンネル登録者数）

## 問題
- subscribersは「データ収集時点」の値
- 動画が人気→チャンネル登録者増加→subscribersが高い
- つまり、結果（views）が原因（subscribers）に影響している

## 質問
1. これは時系列リーケージと言えますか？
2. このような場合、subscribersを特徴量として使うべきではないでしょうか？
3. もし使う場合、どのような前処理が必要でしょうか？
"""

# ファイルに保存
with open('gemini_consultation.txt', 'w', encoding='utf-8') as f:
    f.write(consult_text)

print("\nGeminiへの相談内容を gemini_consultation.txt に保存しました")