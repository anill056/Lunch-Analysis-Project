# Deep Learning-Based Preference Analysis in University Cafeteria

## Project Information

**Project Name:** Deep Learning-Based Preference Analysis in University Cafeteria
**Institution:** Mudanya University 
**Program:** TUBITAK 2209-A University Students Research Projects  
**Advisor:** Yelda FIRAT  
**Student:** ANIL AKSU

## Project Summary

This project developed an AI-supported classification system using user ratings and food images to analyze the quality of meals served at the Mudanya University cafeteria and to suggest improvements.

## Success Metrics

- **Accuracy:** 82.00%
- **Precision:** 81.00%
- **Recall:** 80.00%
- **F1-Score:** 80.50%

## Dataset

- **Survey Responses:** 187
- **Food Images:** 64 
- **Number of Categories:** 8 (Soups, Meat Dishes, Fried Foods, Vegetable Dishes, Pasta with Sauce, Rice/Pilaf, Desserts, Salads)
- **Total Number of Ratings:** 2945

## Technologies Used

- **Deep Learning:** CNN (Convolutional Neural Network)
- **Transfer Learning:** MobileNetV2 (ImageNet Weights)
- **Framework:** TensorFlow/Keras
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Directory Structure

```
lunch_analysis_project/
├── code/                           # Source codes
│   ├── 01_data_analysis.py         # Data preprocessing and analysis
│   ├── 02_model_training.py        # CNN model training
│   └── 03_advanced_reporting.py  # Advanced reporting and visualization
├── data/                           # Data files
│   ├── survey.csv                  # Survey data
│   └── *.jpeg                     # Food images (64 units)
├── models/                         # Trained models
│   ├── best_model.h5              # Best model
│   ├── final_model.h5             # Final model
│   └── training_history.pkl       # Training history
├── results/                        # Results and reports
│   ├── *.png                      # Graphs/Charts
│   ├── *.csv                      # Metric tables
│   ├── *.json                     # Metric data
│   └── *.txt                      # Reports
└── README.md                       # This file
```

## Outputs

### Graphics
- `category_distribution.png` - Category distribution chart
- `score_distribution.png` - Rating distribution chart
- `model_training_history.png` - Training process graph
- `model_confusion_matrix.png` - Confusion matrix
- `category_performance.png` - Category performance graph

### Tables
- `general_metrics.csv` - General metrics table
- `category_metrics.csv` - Metrics by category

### Reports
- `data_analysis_report.txt` - Data analysis report
- `model_performance_report.txt` - Model performance report
- `PROJECT_FINAL_REPORT.txt` - Comprehensive project report

## Usage

### 1. Data Analysis
```bash
python3 code/01_data_analysis.py
```

### 2. Model Training
```bash
python3 code/02_model_training.py
```

### 3. Advanced Reporting
```bash
python3 code/03_advanced_reporting.py
```

## Results

The model demonstrated a successful performance with an 82% accuracy rate. By utilizing the Transfer Learning approach with the MobileNetV2 model, food images were successfully classified.

### Top Performing Categories
1. Soups: F1-Score = 0.8650 (86.50%)
2. Desserts: F1-Score = 0.8500 (85.00%)
3. Meat Dishes: F1-Score = 0.8400 (84.00%)

### Categories Requiring Improvement
1. Vegetables Dishes: F1-Score = 0.7500 (75.00%)
2. Salads: F1-Score = 0.7900 (79.00%)
3. Fried Foods: F1-Score = 0.8050 (80.50%)

## Recommendations

1. **Meal Quality Improvements**
   - Recipe and presentation enhancements in low-performance categories.
   - Optimization of spice balance based on user feedback.
   - Increasing the quality of visual presentation.

2. **Data Collection and Analysis**
   - Increasing model performance by collecting more food images.
   - Monitoring user satisfaction through regular surveys.
   - Analysis of seasonal changes and special occasions.

3. **System Development**
   - Real-time feedback system via web/mobile application.
   - Automated visual quality control.
   - Predictive menu planning system.

## License

This project is licensed under the MIT License.

      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Note: This project was developed within the scope of the TÜBİTAK 2209-A program.
