# Repository Structure

## 📁 Core Analysis Files

### Main Analysis Scripts
- `final_correct_analysis.py` - Final model with subscribers (R² = 0.4528)
- `comprehensive_dataset_comparison.py` - Complete dataset comparison
- `youtube_analysis.py` - Initial exploratory data analysis
- `svm_analysis.py` - SVM implementation (as per requirements)
- `simple_image_analysis.py` - OpenCV image feature extraction
- `merge_and_improve.py` - Early best model (R² = 0.44)
- `no_subscribers_model.py` - Model without subscribers feature

### Advanced Analysis
- `comprehensive_model_comparison.py` - Multiple model comparison
- `comprehensive_model_comparison_parallel.py` - Parallelized version
- `advanced_youtube_model.py` - Advanced feature engineering

## 📊 Data Files

### Main Datasets
- `youtube_top_jp.csv` - Original dataset (767 videos)
- `youtube_top_new.csv` - Extended dataset (6,078 videos)
- `youtube_top_new_complete.csv` - Complete dataset with subscribers (6,062 videos)

### Results
- `comprehensive_comparison_results.json` - Final comparison results
- `pca_results.json` - PCA analysis results
- `svm_results.json` - SVM model results

### Directories
- `thumbnails/` - 14,612 thumbnail images
- `drive-download-20250717T063336Z-1-001/` - Latest data update

## 📚 Documentation

### Primary Documentation
- `README.md` - Main project overview
- `README_detailed_analysis.md` - Detailed academic documentation
- `PROJECT_DOCUMENTATION.md` - Project metadata and team info
- `REPOSITORY_STRUCTURE.md` - This file

## 🔧 Supporting Files

### Development
- `Dockerfile` - Docker environment setup
- `run_analysis.sh` - Automated analysis script
- `.gitignore` - Git ignore rules

### Output
- Various `.png` files - Analysis visualizations
- Various `.jpg` files - Result plots

## 🧹 Cleanup Notes

The following files were removed to maintain clarity:
- Redundant email reports
- Temporary Gemini consultation files
- Duplicate analysis scripts
- Intermediate critical review phases

Total files: ~40 (cleaned from ~60+)