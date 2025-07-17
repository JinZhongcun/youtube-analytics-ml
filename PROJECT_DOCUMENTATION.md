# Project Documentation

## 1. Course Information
- **University**: Osaka University
- **Course**: Group 4 - International Integration Science / Advanced Integration Science
- **Assignment**: Assignment 2
- **Academic Year**: 2025
- **Submission Deadline**: July 24, 2025

## 2. Team Members
- Nakamura Jin
- Meng Siyuan (Data collection lead)
- Haruto Ohto  
- Ryotaro Hada

## 3. Assignment Requirements
1. Use Principal Component Analysis (PCA) to identify key factors
2. Use Support Vector Machine (SVM) for prediction
3. Analyze factors affecting YouTube video views
4. Provide actionable insights for content creators
5. Submit 4-page academic report

## 4. Data Collection Timeline

### Phase 1: Initial Dataset (July 2025)
- **Source**: Kaggle YouTube dataset
- **Size**: 767 videos
- **Features**: Basic metadata + subscribers

### Phase 2: Data Expansion (July 15, 2025)
- **Lead**: Meng Siyuan
- **Size**: 6,078 videos + 14,612 thumbnail images
- **Issue**: 90% missing subscribers data

### Phase 3: Data Completion (July 17, 2025)
- **Update**: Meng Siyuan restored subscribers data
- **Size**: 6,062 videos with complete features
- **Result**: Final R² = 0.4528

## 5. Key Collaboration Points

1. **Initial Problem**: Low performance (R² = 0.21) due to limited data
2. **Solution**: Meng Siyuan collected extensive dataset with thumbnails
3. **Challenge**: Missing subscribers column in 90% of new data
4. **Resolution**: Created multiple models and eventually got complete data

## 6. Data Fields Documentation

### Features Used in Final Model
1. **video_duration** - Video length in seconds
2. **tags_count** - Number of tags
3. **description_length** - Description text length
4. **subscribers** - Channel subscriber count
5. **brightness** - Thumbnail brightness (0-255)
6. **colorfulness** - Color diversity metric
7. **object_complexity** - YOLO detected objects
8. **element_complexity** - Visual elements count
9. **hour_published** - Upload hour (0-23)
10. **weekday_published** - Upload day (0-6)

### Features Intentionally Excluded
1. **likes** - Post-view engagement (data leakage)
2. **comment_count** - Post-view engagement (data leakage)
3. **subscriber_per_view** - Contains target variable
4. **title** - Text analysis not implemented

### Data Leakage Prevention
- Excluded all features that are outcomes of viewing
- Only used features available at upload time
- Carefully validated temporal consistency

## 7. Methodology Summary

### Image Feature Extraction (OpenCV)
```python
- Face detection: Haar Cascade
- Color analysis: HSV color space
- Edge detection: Canny algorithm
- Text detection: EAST text detector
```

### Model Selection
- **Primary**: LightGBM (best performance)
- **Alternatives**: XGBoost, Random Forest
- **Baseline**: Linear Regression

### Evaluation Method
- 5-fold Cross Validation
- Train/Test split: 80/20
- Metric: R² (coefficient of determination)

## 8. Acknowledgments

Special thanks to **Meng Siyuan** for:
- Collecting 6,078 videos with thumbnails
- Fixing the subscribers data issue
- Enabling the project's success with data engineering