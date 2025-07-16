# Analysis Methodology Documentation

## Feature Engineering Details

### Temporal Features Created
1. **days_since_publish** - Calculated from current date - published_at
2. **hour_published** - Hour of day when video was published (0-23)
3. **weekday_published** - Day of week (0=Monday, 6=Sunday)

### Log Transformations Applied
1. **log_views** - log10(views + 1) as target variable
2. **log_duration** - log10(video_duration + 1)
3. **log_tags** - log10(tags_count + 1)
4. **log_desc_length** - log10(description_length + 1)
5. **log_subscribers** - log10(subscribers + 1) when available

### Engineered Ratios
1. **tags_per_second** - tags_count / (video_duration + 1)
2. **desc_per_second** - description_length / (video_duration + 1)

## Image Analysis Methods

### OpenCV Feature Extraction Process
1. **Color Space Analysis**
   - Convert BGR to HSV
   - Calculate mean and std for each channel
   - Compute colorfulness as std(HSV)

2. **Face Detection**
   - Used Haar Cascade classifier
   - haarcascade_frontalface_default.xml
   - Binary feature: has_face (0 or 1)
   - Count feature: num_faces

3. **Edge Detection**
   - Canny edge detector (thresholds: 50, 150)
   - Edge density = edge_pixels / total_pixels

4. **Text Area Estimation**
   - Binary threshold at 200
   - text_area_ratio = white_pixels / total_pixels

5. **Advanced Features**
   - LAB color space for vibrancy
   - Quadrant analysis for composition
   - Center brightness for focus area

## Model Selection Rationale

### Why These Models?
1. **LightGBM** - Fast, handles categorical features well
2. **XGBoost** - Robust to overfitting, good with small data
3. **Random Forest** - Baseline comparison, interpretable
4. **Ensemble** - Voting regressor to combine strengths

### Why Not Deep Learning?
- Dataset too small (607-6,078 samples)
- Risk of overfitting
- Classical ML achieved good results (R² = 0.44)
- Interpretability was important for insights

## Validation Strategy

### Train-Test Split
- 80-20 split
- Random state = 42 for reproducibility
- Stratified by view count quantiles

### No Cross-Validation Because
- Limited data (607 complete records)
- Computational efficiency needed
- Clear train-test split sufficient

## Performance Metrics

### Primary Metric: R² Score
- Measures proportion of variance explained
- Easy to interpret (0-1 scale)
- Standard for regression tasks

### Secondary Metrics Considered
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Key Decisions and Rationale

### 1. Excluding Engagement Metrics
- **Decision**: Remove likes, comments from features
- **Rationale**: Prevent data leakage, ensure realistic predictions

### 2. Handling Missing Subscribers
- **Decision**: Two separate models (with/without)
- **Rationale**: Use all available data effectively

### 3. Image Feature Priority
- **Decision**: Extract 15+ image features
- **Rationale**: Compensate for missing subscriber data

### 4. Log Transform Views
- **Decision**: Predict log10(views) not raw views
- **Rationale**: Handle exponential distribution, improve model stability

## Limitations Acknowledged

1. **Temporal Bias**: Older videos have more time to accumulate views
2. **Survivor Bias**: Only analyzing successful (trending) videos
3. **Geographic Bias**: Japan-focused dataset
4. **Feature Completeness**: No audio analysis, limited text analysis

## Reproducibility Measures

1. **Fixed Random Seeds**: random_state=42 throughout
2. **Docker Environment**: Consistent dependencies
3. **Data Versioning**: Saved CSV files with analysis
4. **Code Documentation**: Extensive comments in scripts

## Future Improvements Identified

1. **Collect Missing Data**: YouTube API for subscribers
2. **Text Analysis**: NLP on titles and descriptions
3. **Temporal Modeling**: Time series for view growth
4. **A/B Testing**: Validate findings with creators
5. **CNN Implementation**: If more data available