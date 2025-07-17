#!/usr/bin/env python3
"""
批判的査読 Phase 1: データ整合性の徹底検証
疑問点：
1. なぜ元データ6078件→新データ6062件に減った？
2. viewsが98%も変わっているのはなぜ？
3. subscribersは本当に「現在」の値か？
4. データ収集時期の不整合はないか？
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("批判的査読 Phase 1: データ整合性の徹底検証")
print("="*80)

# データ読み込み
df_old = pd.read_csv('youtube_top_new.csv')
df_new = pd.read_csv('youtube_top_new_complete.csv')

print("\n【1. データ件数の不整合】")
print(f"元データ: {len(df_old)}件")
print(f"新データ: {len(df_new)}件")
print(f"差分: {len(df_old) - len(df_new)}件が消失")

# 消えたvideo_idを特定
old_ids = set(df_old['video_id'])
new_ids = set(df_new['video_id'])
missing_ids = old_ids - new_ids
print(f"\n消失したvideo_id: {len(missing_ids)}件")
if len(missing_ids) > 0:
    print("例:", list(missing_ids)[:5])

# マージして詳細比較
merged = pd.merge(df_old, df_new, on='video_id', suffixes=('_old', '_new'), how='inner')
print(f"\n【2. 共通video_idの詳細比較】")
print(f"マージできた件数: {len(merged)}件")

# viewsの変化を分析
merged['views_diff'] = merged['views_new'] - merged['views_old']
merged['views_diff_pct'] = (merged['views_diff'] / merged['views_old'] * 100)

print("\n【3. viewsの変化分析】")
print(f"viewsが増加: {(merged['views_diff'] > 0).sum()}件 ({(merged['views_diff'] > 0).sum()/len(merged)*100:.1f}%)")
print(f"viewsが減少: {(merged['views_diff'] < 0).sum()}件 ({(merged['views_diff'] < 0).sum()/len(merged)*100:.1f}%)")
print(f"viewsが不変: {(merged['views_diff'] == 0).sum()}件")

# views変化の統計
print("\nviews変化率の統計:")
print(f"平均: {merged['views_diff_pct'].mean():.2f}%")
print(f"中央値: {merged['views_diff_pct'].median():.2f}%")
print(f"最大増加: {merged['views_diff_pct'].max():.2f}%")
print(f"最大減少: {merged['views_diff_pct'].min():.2f}%")

# 時間的整合性チェック
print("\n【4. 時間的整合性の検証】")
merged['published_at_old'] = pd.to_datetime(merged['published_at_old'])
merged['published_at_new'] = pd.to_datetime(merged['published_at_new'])
merged['published_diff'] = (merged['published_at_new'] - merged['published_at_old']).dt.total_seconds()

print(f"published_atが変わった動画: {(merged['published_diff'] != 0).sum()}件")

# time_durationの検証
print("\n【5. time_durationの検証】")
print(f"time_duration_oldの範囲: {merged['time_duration_old'].min()} - {merged['time_duration_old'].max()}")
print(f"time_duration_newの範囲: {merged['time_duration_new'].min()} - {merged['time_duration_new'].max()}")

# time_durationと日数の関係を確認
merged['published_at_old_tz'] = merged['published_at_old'].dt.tz_localize(None)
merged['calc_days'] = (pd.Timestamp.now().tz_localize(None) - merged['published_at_old_tz']).dt.days
merged['time_duration_check'] = abs(merged['time_duration_old'] - merged['calc_days'])
print(f"\ntime_durationが経過日数と一致しない動画: {(merged['time_duration_check'] > 1).sum()}件")

# subscribersとviewsの時系列関係
print("\n【6. subscribersの時系列整合性】")
# 古い動画ほどsubscribersが多いはず（チャンネルが成長するため）
merged['days_old'] = (pd.Timestamp.now() - merged['published_at_old']).dt.days
correlation = merged[['days_old', 'subscribers']].corr().iloc[0, 1]
print(f"経過日数とsubscribersの相関: {correlation:.4f}")
print("（正の相関なら時系列的に矛盾）")

# データの重複チェック
print("\n【7. データの重複・異常値】")
print(f"video_idの重複（元データ）: {df_old['video_id'].duplicated().sum()}")
print(f"video_idの重複（新データ）: {df_new['video_id'].duplicated().sum()}")

# サムネイル特徴量の整合性
print("\n【8. サムネイル特徴量の整合性】")
thumbnail_cols = ['brightness', 'colorfulness', 'object_complexity', 'element_complexity']
for col in thumbnail_cols:
    if col + '_old' in merged.columns and col + '_new' in merged.columns:
        diff = abs(merged[col + '_new'] - merged[col + '_old'])
        changed = (diff > 0.01).sum()
        print(f"{col}が変化: {changed}件 ({changed/len(merged)*100:.1f}%)")

# 疑わしいデータのサンプル
print("\n【9. 疑わしいデータの例】")
# viewsが激増した動画
suspicious = merged.nlargest(5, 'views_diff_pct')[['video_id', 'title_old', 'views_old', 'views_new', 'views_diff_pct', 'subscribers']]
print("\nviews激増TOP5:")
print(suspicious)

# 結論
print("\n【批判的結論】")
print("1. データの時期が異なる（viewsが更新されている）")
print("2. subscribersは「データ取得時点」の値で、動画公開時ではない")
print("3. time_durationは現在からの経過日数")
print("4. これらの時系列不整合が高いR²の原因かもしれない")

# CSVに保存して詳細確認用
merged[['video_id', 'views_old', 'views_new', 'views_diff_pct', 'subscribers', 
        'days_old', 'published_at_old']].to_csv('critical_review_data.csv', index=False)