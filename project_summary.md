# YouTube Analytics ML Project - Comprehensive Summary

## Project Overview
**Assignment 2**: Identify factors affecting YouTube video views using PCA and SVM to help content creators increase viewership.

## Timeline & Key Milestones

### Initial Phase
- Started with 767 YouTube videos dataset (`youtube_top_jp.csv`)
- Initial model achieved R² = 0.21
- Identified data limitation: too small for deep learning approaches

### Data Enhancement Phase
- Received 6,078 videos with thumbnail images from teammate Meng Siyuan
- Downloaded and extracted thumbnail images from Google Drive
- Merged datasets on video_id, resulting in 607 videos with complete data

### Model Development Phase
- Implemented comprehensive image analysis using OpenCV
- Compared 7+ ML models (Random Forest, XGBoost, LightGBM, SVM, etc.)
- Achieved final R² = 0.4416 (110% improvement)

## Technical Implementation

### Data Processing
```python
# Merging old and new data
df_merged = pd.merge(df_new, df_old[['video_id', 'subscribers']], 
                     on='video_id', how='left')
```

### Image Feature Extraction
- Face detection using Haar Cascades
- Edge density calculation
- Color analysis (HSV space)
- Brightness and colorfulness metrics

### Key Features Identified
1. **colorfulness** (0.226) - Most important
2. **video_duration** (0.207) 
3. **subscribers** (0.199)
4. **tags_count** (0.170)
5. **object_complexity** (0.168)

## Results

### Model Performance
| Model | R² Score | Improvement |
|-------|----------|-------------|
| Random Forest + Images | 0.4416 | +110% |
| Ensemble | 0.4334 | +106% |
| XGBoost + Images | 0.4103 | +95% |
| Initial Model | 0.2102 | - |

### Key Findings
1. **Short videos perform better**: 90 seconds vs 689 seconds (-87%)
2. **Colorful thumbnails** are most effective (but not too flashy)
3. **Faces in thumbnails** surprisingly perform worse
4. **5M subscribers** is the critical threshold

## Recommendations for Content Creators
1. Keep videos under 90 seconds
2. Use moderately colorful thumbnails (brightness ~70)
3. Focus on content visualization over faces
4. Build subscriber base to 5M+ for viral potential

## Project Deliverables
- GitHub Repository: https://github.com/JinZhongcun/youtube-analytics-ml
- All analysis code and data
- Comprehensive documentation
- Final report email (bilingual)

## Technical Stack
- Python (pandas, scikit-learn, OpenCV)
- LightGBM, XGBoost
- Docker for reproducibility
- Parallel processing (32 CPUs)

## Files Created
- `youtube_analysis.py` - EDA and PCA analysis
- `svm_analysis.py` - SVM implementation
- `simple_image_analysis.py` - Image feature extraction
- `merge_and_improve.py` - Data integration and best model
- `comprehensive_model_comparison_parallel.py` - All model comparisons
- Various visualization outputs

## Lessons Learned
1. **Data quality > quantity**: 607 videos with complete features outperformed 6,078 with partial data
2. **Feature engineering matters**: Image features provided 2x improvement
3. **Simple models work**: With limited data, ensemble of simple models beats complex architectures
4. **Domain knowledge helps**: Understanding YouTube's algorithm (subscribers importance) was crucial

## Next Steps (Suggested)
1. Collect more data with subscriber information
2. Implement time-series features
3. Category-specific models
4. A/B testing recommendations with real creators