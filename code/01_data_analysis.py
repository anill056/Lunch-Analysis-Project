#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Food Analysis Project - Data Preprocessing and Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For character support
plt.rcParams['font.family'] = 'DejaVu Sans'

# Define directories
BASE_DIR = Path('/home/ubuntu/lunch_analysis_project')
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

def load_survey_data():
    """Load and clean survey data"""
    print("Loading survey data...")
    df = pd.read_csv(DATA_DIR / 'survey.csv')
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    return df

def analyze_categories(df):
    """Analyze food categories"""
    print("\n=== Category Analysis ===")
    
    # Second column contains category information
    category_column = df.columns[1]
    
    # Categorize counts
    categories = {
        'Çorbalar':  0,
        'Etli yemekler': 0,
        'Kızartmalar': 0,
        'Sebze yemekleri': 0,
        'Soslu makarnalar': 0,
        'Pilavlar': 0,
        'Tatlılar': 0,
        'Salatalar': 0
    }

    
    for idx, row in df.iterrows():
        category_text = str(row[category_column])
        if pd.notna(category_text) and category_text != 'nan':
            for category in categories.keys():
                if category in category_text or category.lower() in category_text.lower():
                    categories[category] += 1
    
    # Convert results to DataFrame
    category_df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
    category_df = category_df[category_df['Count'] > 0].sort_values('Count', ascending=False)

    
    print("\nCategory Distribution:")
    print(category_df.to_string(index=False))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(category_df['Category'], category_df['Count'], color='steelblue', alpha=0.8)
    plt.xlabel('Food Category', fontsize=12)
    plt.ylabel('Survey Count', fontsize=12)
    plt.title('Survey Distribution by Food Categories', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'category_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved: {RESULTS_DIR / 'category_distribution.png'}")
    plt.close()
    
    return category_df

def analyze_scores(df):
    """Analyze rating data"""
    print("\n=== Score Analysis ===")
    
    # Find numeric columns (ratings)
    numeric_cols = []
    for col in df.columns[2:]:  # First two columns are Date and Category
        try:
            # Try to convert column to numeric
            pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)
        except:
            pass

    # Calculate the average of all ratings
    score_list = []
    for col in df.columns[2:]:
        try:
            score_series = pd.to_numeric(df[col], errors='coerce')
            valid_scores = score_series.dropna()
            if len(valid_scores) > 0:
                score_list.extend(valid_scores.tolist())
        except:
            pass
    
    if score_list:
        score_array = np.array(score_list)
        score_array = score_array[(score_array >= 1) & (score_array <= 5)]  # Scores between 1-5
        
        print(f"\nTotal score count: {len(score_array)}")
        print(f"Average score: {np.mean(score_array):.2f}")
        print(f"Median score: {np.median(score_array):.2f}")
        print(f"Standard deviation: {np.std(score_array):.2f}")
        print(f"Minimum score: {np.min(score_array):.1f}")
        print(f"Maksimum score: {np.max(score_array):.1f}")
        
        # Score distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(score_array, bins=np.arange(0.5, 6, 0.5), color='coral', alpha=0.7, edgecolor='black')
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('General Score Distribution', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(1, 6))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'score_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {RESULTS_DIR / 'score_distribution.png'}")
        plt.close()
        
        # Score statistics
        score_stats = {
            'Metric': ['Average', 'Median', 'Std Dev', 'Minimum', 'Maximum', 'Total Score Count'],
            'Value': [
                f"{np.mean(score_array):.2f}",
                f"{np.median(score_array):.2f}",
                f"{np.std(score_array):.2f}",
                f"{np.min(score_array):.1f}",
                f"{np.max(score_array):.1f}",
                f"{len(score_array)}"
            ]
        }
        stats_df = pd.DataFrame(score_stats)
        print("\nScore Statistics:")
        print(stats_df.to_string(index=False))
        
        return score_array, stats_df
    
    return None, None

def analyze_visual_data():
    """Analyze visual dataset"""
    print("\n=== Visual Data Analysis ===")
    
    # Count image files
    image_files = list(DATA_DIR.glob('*.jpeg')) + list(DATA_DIR.glob('*.jpg'))
    print(f"Total number of images: {len(image_files)}")
    
    # Analyze file sizes
    file_sizes = [f.stat().st_size / 1024 for f in image_files]  # In KB
    
    print(f"Average file size: {np.mean(file_sizes):.2f} KB")
    print(f"Minimum file size: {np.min(file_sizes):.2f} KB")
    print(f"Maximum file size: {np.max(file_sizes):.2f} KB")
    
    visual_stats = {
        'Metric': ['Total Images', 'Avg Size (KB)', 'Min Size (KB)', 'Max Size (KB)'],
        'Value': [
            len(image_files),
            f"{np.mean(file_sizes):.2f}",
            f"{np.min(file_sizes):.2f}",
            f"{np.max(file_sizes):.2f}"
        ]
    }
    visual_df = pd.DataFrame(visual_stats)
    print("\nVisual Statistics:")
    print(visual_df.to_string(index=False))
    
    return len(image_files), visual_df

def create_summary_report(category_df, score_array, stats_df, visual_count, visual_df):
    """Create summary report"""
    print("\n=== Creating Summary Report ===")
    
    report = []
    report.append("# FOOD ANALYSIS PROJECT - DATA ANALYSIS REPORT\n")
    report.append("=" * 60 + "\n\n")
    
    report.append("## 1. GENERAL INFORMATION\n")
    report.append(f"- Project Name: Deep Learning-Based Preference Analysis in University Cafeteria\n")
    report.append(f"- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## 2. DATASET STATISTICS\n\n")
    
    report.append("### 2.1 Category Distribution\n")
    report.append(category_df.to_string(index=False))
    report.append("\n\n")
    
    if stats_df is not None:
        report.append("### 2.2 Score Statistics\n")
        report.append(stats_df.to_string(index=False))
        report.append("\n\n")
    
    report.append("### 2.3 Visual Data Statistics\n")
    report.append(visual_df.to_string(index=False))
    report.append("\n\n")
    
    report.append("## 3. RESULTS\n\n")
    report.append("Dataset successfully analyzed. There are {} survey responses and {} images across {} categories.\n".format(
        category_df['Count'].sum(), visual_count, len(category_df)
    ))
    
    if score_array is not None:
        report.append(f"Overall satisfaction average: {np.mean(score_array):.2f}/5.00\n\n")
    
    report.append("## 4. GENERATED OUTPUTS\n\n")
    report.append("- category_distribution.png\n")
    report.append("- score_distribution.png\n")
    report.append("- data_analysis_report.txt\n\n")
    
    # Save report
    with open(RESULTS_DIR / 'data_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"Report saved: {RESULTS_DIR / 'data_analysis_report.txt'}")

def main():
    """Main function"""
    print("=" * 60)
    print("FOOD ANALYSIS PROJECT - DATA ANALYSIS")
    print("=" * 60)
    
    # Load survey data
    df = load_survey_data()
    
    # Category analysis
    category_df = analyze_categories(df)
    
    # Score analysis
    score_array, stats_df = analyze_scores(df)
    
    # Visual data analysis
    visual_count, visual_df = analyze_visual_data()
    
    # Summary report
    create_summary_report(category_df, score_array, stats_df, visual_count, visual_df)
    
    print("\n" + "=" * 60)
    print("DATA ANALYSIS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

