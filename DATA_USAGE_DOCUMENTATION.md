# Data Usage Documentation

## Data Fields Available in Dataset

### youtube_top_jp.csv (767 videos)
- video_id
- title
- category_id
- video_duration
- tags_count
- views
- **likes** ✗
- **comment_count** ✗
- thumbnail_link
- description_length
- time_duration
- published_at
- **subscribers** ✓
- keyword
- thumbnail_path
- object_complexity
- element_complexity
- brightness
- colorfulness

### youtube_top_new.csv (6,078 videos)
- video_id
- title
- category_id
- video_duration
- tags_count
- views
- thumbnail_link
- description_length
- time_duration
- published_at
- keyword
- thumbnail_path
- object_complexity
- element_complexity
- brightness
- colorfulness
- **No subscribers**
- **No likes**
- **No comment_count**

## Data Used in Models

### Features USED (✓)
1. **video_duration** - Length of video in seconds
2. **tags_count** - Number of tags
3. **description_length** - Length of description text
4. **subscribers** - Channel subscriber count (when available)
5. **object_complexity** - Thumbnail complexity score
6. **element_complexity** - Visual element complexity
7. **brightness** - Thumbnail brightness value
8. **colorfulness** - Thumbnail color diversity
9. **category_id** - Video category (one-hot encoded)
10. **published_at** - Converted to days_since_publish
11. **Image features** (extracted using OpenCV):
    - hue_mean, hue_std
    - saturation_mean, saturation_std
    - value_mean, value_std
    - edge_density
    - has_face, num_faces
    - text_area_ratio
    - color_diversity
    - center_brightness
    - color_vibrancy
    - contrast
    - quadrant_brightness_std

### Features EXCLUDED (✗)
1. **likes** - Intentionally excluded (data leakage)
2. **comment_count** - Intentionally excluded (data leakage)
3. **dislike_count** - Not available in dataset
4. **video_id** - Just identifier, not predictive
5. **title** - Text analysis not implemented
6. **thumbnail_link** - URL not useful for prediction
7. **keyword** - Redundant with category

## Reason for Exclusion

### Data Leakage Prevention
**likes** and **comment_count** were intentionally excluded because:
- They are outcomes that occur AFTER viewing
- Including them would create unrealistic model performance
- At prediction time (before upload), these values don't exist
- As Meng Siyuan correctly noted: "they're outcomes of popularity, not causes of it"

### Model Configurations

#### Configuration 1: Complete Data (607 videos)
- Used ALL features including subscribers
- Result: R² = 0.44 (best performance)

#### Configuration 2: No Subscribers (6,078 videos)
- Excluded subscribers feature
- Used all other features
- Result: R² = 0.34 (practical alternative)

#### Configuration 3: Initial Model (767 videos)
- Limited features (no image analysis)
- Result: R² = 0.21 (baseline)

## Data Quality Issues

### Missing Data Problem
- 5,471 videos (90%) missing subscribers data
- This critical feature absence limited full dataset usage
- Merging on video_id only yielded 607 complete records

### Solution Attempts
1. Used only complete data (607 videos) → Best results
2. Built model without subscribers → Acceptable results
3. Future: Collect missing subscribers via YouTube API