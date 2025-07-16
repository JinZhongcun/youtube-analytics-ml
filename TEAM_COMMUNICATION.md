# Team Communication Log

## Final Project Summary (2025/07/16)

### 日本語版

データが不足していて結果が良くなかったので、@MENG SIYUAN さんにめっちゃデータ拡張をしていただきました。（感謝！）

拡張データには重要だったsubscribersデータに欠損が確認されたため、いくつかの実験を行うことにしました。

欠損なしのみ（10%程度のみ使用）での結果が一番よく、subscribersデータを使わない（100%のデータを使用）の場合も悪くない結果が出ました。

これでいいんじゃないかなと思ってますが、より良くするには、すべてのデータにsubscribersを加えることかなと思ってます。

### English Version

Because of the lack of the data, I beg @MENG SIYUAN to get more data, and he did. (Thank you!)

I did some experiments due to the configuration of the lack of the column of "subscribers", which is important to predict.

The condition 1 (Using only 10% that data has "subscribers" data) is the most good accuracy, and the condition2 (100% data without "subscribers" data) is the secondly good.

I think our mission is completed, but if you think we want to make more good model, we should use the data which has "subscriber" data.

## Experimental Results Summary

| Condition | Data Used | Key Features | R² Score | Notes |
|-----------|-----------|--------------|----------|-------|
| Condition 1 | 607 videos (10%) | All features including subscribers | **0.44** | Best accuracy |
| Condition 2 | 6,078 videos (100%) | Without subscribers | 0.34 | Good alternative |
| Initial | 767 videos | Basic features only | 0.21 | Poor performance |

## Key Collaboration Points

1. **Data Expansion**: Meng Siyuan provided 6,078 videos with thumbnail images
2. **Problem Identified**: 90% of new data missing subscribers column
3. **Solution**: Created two models to handle both scenarios
4. **Future Improvement**: Add subscribers data to all 6,078 videos via YouTube API

## Acknowledgments

Special thanks to **Meng Siyuan** for the massive data collection effort that made this project successful!