import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_metrics_file(filepath):
    """解析单个metrics文件"""
    if not os.path.exists(filepath):
        return None
    
    metrics = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取患者ID从文件名
        patient_id = os.path.basename(filepath).replace('_metrics.txt', '')
        metrics['patient_id'] = patient_id
        
        # 解析整体指标 (OVERALL METRICS部分)
        if "OVERALL METRICS:" in content:
            overall_section = content.split("OVERALL METRICS:")[1]
            if "SLICE-LEVEL METRICS:" in overall_section:
                overall_section = overall_section.split("SLICE-LEVEL METRICS:")[0]
            elif "DETAILED SLICE METRICS:" in overall_section:
                overall_section = overall_section.split("DETAILED SLICE METRICS:")[0]
            
            for line in overall_section.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        metrics[key] = float(value)
                    except:
                        pass
        
        # 解析切片级别平均指标 (SLICE-LEVEL METRICS部分)
        if "SLICE-LEVEL METRICS:" in content:
            slice_section = content.split("SLICE-LEVEL METRICS:")[1]
            if "DETAILED SLICE METRICS:" in slice_section:
                slice_section = slice_section.split("DETAILED SLICE METRICS:")[0]
            
            for line in slice_section.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        metrics[key] = float(value)
                    except:
                        pass
        
        # 解析统计信息
        stats_patterns = {
            'total_slices': r"Total slices:\s+(\d+)",
            'slices_with_tumor': r"Slices with tumor:\s+(\d+)",
            'predicted_tumors': r"Predicted tumors:\s+(\d+)",
            'true_tumor_volume': r"True tumor volume:\s+(\d+)",
            'pred_tumor_volume': r"Pred tumor volume:\s+(\d+)"
        }
        
        for key, pattern in stats_patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = int(match.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def find_all_metrics_files(directory):
    """查找所有metrics文件"""
    metrics_files = []
    
    # 在目录中搜索所有_metrics.txt文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_metrics.txt'):
                metrics_files.append(os.path.join(root, file))
    
    print(f"Found {len(metrics_files)} metrics files")
    return sorted(metrics_files)

def calculate_summary_statistics(all_metrics):
    """计算汇总统计"""
    if not all_metrics:
        return {}
    
    # 转换为DataFrame以便分析
    df = pd.DataFrame(all_metrics)
    
    # 定义要分析的指标
    metric_columns = [
        'accuracy', 'dice', 'iou', 'precision', 'recall', 
        'f1_score', 'specificity', 'volume_error', 'relative_volume_diff'
    ]
    
    # 过滤存在的列
    existing_columns = [col for col in metric_columns if col in df.columns]
    
    summary_stats = {}
    
    for col in existing_columns:
        # 去除NaN值
        values = df[col].dropna()
        if len(values) > 0:
            summary_stats[col] = {
                'count': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            }
    
    return summary_stats, df

def generate_summary_report(all_metrics, summary_stats, df, output_dir):
    """生成汇总报告"""
    report_path = os.path.join(output_dir, "metrics_summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BRAIN TUMOR SEGMENTATION - METRICS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        f.write(f"SUMMARY STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total patients analyzed: {len(all_metrics)}\n")
        f.write(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 主要性能指标
        f.write("KEY PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        
        # 创建格式化的表格
        header = f"{'Metric':<15} {'Count':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score']:
            if metric in summary_stats:
                stats = summary_stats[metric]
                f.write(f"{metric.upper():<15} {stats['count']:<6} "
                       f"{stats['mean']:<8.4f} {stats['std']:<8.4f} "
                       f"{stats['min']:<8.4f} {stats['max']:<8.4f} "
                       f"{stats['median']:<8.4f}\n")
        
        f.write("\n")
        
        # 性能分级
        if 'dice' in summary_stats:
            dice_mean = summary_stats['dice']['mean']
            if dice_mean >= 0.9:
                grade = "EXCELLENT (A+)"
            elif dice_mean >= 0.8:
                grade = "VERY GOOD (A)"
            elif dice_mean >= 0.7:
                grade = "GOOD (B)"
            elif dice_mean >= 0.6:
                grade = "FAIR (C)"
            else:
                grade = "NEEDS IMPROVEMENT (D)"
            
            f.write("OVERALL PERFORMANCE ASSESSMENT\n")
            f.write("-" * 35 + "\n")
            f.write(f"Model Performance Grade: {grade}\n")
            f.write(f"Average Dice Coefficient: {dice_mean:.4f} ± {summary_stats['dice']['std']:.4f}\n")
            
            if 'iou' in summary_stats:
                f.write(f"Average IoU Score: {summary_stats['iou']['mean']:.4f} ± {summary_stats['iou']['std']:.4f}\n")
            
            f.write("\n")
        
        # 体积分析
        if 'true_tumor_volume' in df.columns and 'pred_tumor_volume' in df.columns:
            total_true = df['true_tumor_volume'].sum()
            total_pred = df['pred_tumor_volume'].sum()
            
            f.write("VOLUME ANALYSIS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total true tumor volume: {total_true:,} voxels\n")
            f.write(f"Total predicted volume: {total_pred:,} voxels\n")
            f.write(f"Overall volume ratio (pred/true): {total_pred/total_true:.4f}\n")
            
            if total_pred > total_true * 1.2:
                f.write("⚠️  Model tends to OVER-segment (predicts too much tumor)\n")
            elif total_pred < total_true * 0.8:
                f.write("⚠️  Model tends to UNDER-segment (misses tumor regions)\n")
            else:
                f.write("✅ Model shows balanced volume estimation\n")
            f.write("\n")
        
        # Top和Bottom患者
        if 'dice' in df.columns:
            f.write("PATIENT PERFORMANCE RANKING\n")
            f.write("-" * 30 + "\n")
            
            # Top 10患者
            top_patients = df.nlargest(10, 'dice')[['patient_id', 'dice', 'iou', 'precision', 'recall']]
            f.write("TOP 10 PERFORMERS (by Dice Score):\n")
            f.write(f"{'Rank':<5} {'Patient':<10} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8}\n")
            f.write("-" * 60 + "\n")
            for i, (_, row) in enumerate(top_patients.iterrows(), 1):
                f.write(f"{i:<5} {row['patient_id']:<10} {row['dice']:<8.4f} "
                       f"{row.get('iou', 0):<8.4f} {row.get('precision', 0):<10.4f} "
                       f"{row.get('recall', 0):<8.4f}\n")
            
            f.write("\n")
            
            # 低性能患者
            poor_performers = df[df['dice'] < 0.5]
            if len(poor_performers) > 0:
                f.write(f"LOW PERFORMERS (Dice < 0.5): {len(poor_performers)} patients\n")
                for _, row in poor_performers.iterrows():
                    f.write(f"  - Patient {row['patient_id']}: Dice = {row['dice']:.4f}\n")
                f.write("\n")
        
        # 推荐和建议
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        if 'dice' in summary_stats:
            dice_mean = summary_stats['dice']['mean']
            dice_std = summary_stats['dice']['std']
            
            if dice_mean < 0.7:
                f.write("🔧 Model performance below clinical standards. Consider:\n")
                f.write("   - Increase training data size\n")
                f.write("   - Apply data augmentation\n")
                f.write("   - Tune hyperparameters\n")
                f.write("   - Try different loss functions\n\n")
            
            if dice_std > 0.2:
                f.write("📊 High performance variability. Consider:\n")
                f.write("   - Analyze patient characteristics\n")
                f.write("   - Use stratified training\n")
                f.write("   - Implement ensemble methods\n\n")
            
            poor_ratio = len(df[df['dice'] < 0.5]) / len(df) if 'dice' in df.columns else 0
            if poor_ratio > 0.2:
                f.write("⚠️  >20% patients show poor performance. Consider:\n")
                f.write("   - Review data quality\n")
                f.write("   - Perform cross-validation\n")
                f.write("   - Improve model architecture\n\n")
        
        # 详细患者列表
        f.write("DETAILED PATIENT METRICS\n")
        f.write("-" * 25 + "\n")
        f.write(f"{'Patient':<10} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}\n")
        f.write("-" * 60 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['patient_id']:<10} {row.get('dice', 0):<8.4f} "
                   f"{row.get('iou', 0):<8.4f} {row.get('precision', 0):<10.4f} "
                   f"{row.get('recall', 0):<8.4f} {row.get('f1_score', 0):<8.4f}\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path

def create_visualizations(df, summary_stats, output_dir):
    """创建可视化图表"""
    viz_dir = os.path.join(output_dir, "summary_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. 主要指标分布图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Metrics Distribution Across All Patients', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics_to_plot):
        row, col = i // 3, i % 3
        if metric in df.columns and df[metric].notna().any():
            data = df[metric].dropna()
            axes[row, col].hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                                 label=f'Mean: {data.mean():.3f}')
            axes[row, col].axvline(data.median(), color='green', linestyle='--', linewidth=2,
                                 label=f'Median: {data.median():.3f}')
            axes[row, col].set_title(f'{metric.upper()} Distribution', fontweight='bold')
            axes[row, col].set_xlabel(metric.upper())
            axes[row, col].set_ylabel('Number of Patients')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 患者性能热图
    if len(df) > 1:
        metric_cols = ['dice', 'iou', 'precision', 'recall', 'f1_score']
        available_cols = [col for col in metric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            plt.figure(figsize=(12, max(8, len(df)*0.3)))
            
            # 准备热图数据
            heatmap_data = df[['patient_id'] + available_cols].set_index('patient_id')[available_cols]
            
            # 创建热图
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
            plt.title('Patient Performance Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Metrics', fontweight='bold')
            plt.ylabel('Patients', fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'patient_performance_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. 性能排名条形图
    if 'dice' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # 按Dice得分排序
        sorted_df = df.sort_values('dice', ascending=True)
        
        # 创建颜色映射
        colors = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'lightgreen' if x < 0.8 else 'darkgreen' 
                 for x in sorted_df['dice']]
        
        bars = plt.barh(range(len(sorted_df)), sorted_df['dice'], color=colors, alpha=0.8)
        
        # 添加患者ID标签
        plt.yticks(range(len(sorted_df)), sorted_df['patient_id'])
        plt.xlabel('Dice Score', fontweight='bold')
        plt.ylabel('Patient ID', fontweight='bold')
        plt.title('Patient Performance Ranking (by Dice Score)', fontsize=16, fontweight='bold')
        
        # 添加平均线
        mean_dice = df['dice'].mean()
        plt.axvline(mean_dice, color='blue', linestyle='--', linewidth=2, 
                   label=f'Average: {mean_dice:.3f}')
        
        # 添加性能区间线
        plt.axvline(0.5, color='red', linestyle=':', alpha=0.7, label='Poor (0.5)')
        plt.axvline(0.7, color='orange', linestyle=':', alpha=0.7, label='Fair (0.7)')
        plt.axvline(0.8, color='green', linestyle=':', alpha=0.7, label='Good (0.8)')
        
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'patient_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 指标相关性矩阵
    metric_cols = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    available_cols = [col for col in metric_cols if col in df.columns and df[col].notna().any()]
    
    if len(available_cols) > 2:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[available_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Metrics Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {viz_dir}")

def save_summary_data(all_metrics, summary_stats, df, output_dir):
    """保存汇总数据"""
    # 保存JSON格式
    json_path = os.path.join(output_dir, "metrics_summary.json")
    summary_data = {
        'summary_statistics': summary_stats,
        'patient_count': len(all_metrics),
        'generation_time': pd.Timestamp.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # 保存CSV格式
    csv_path = os.path.join(output_dir, "all_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # 保存Excel格式（如果pandas支持）
    try:
        excel_path = os.path.join(output_dir, "metrics_summary.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Metrics', index=False)
            
            # 创建汇总统计表
            stats_df = pd.DataFrame(summary_stats).T
            stats_df.to_excel(writer, sheet_name='Summary_Statistics')
        
        print(f"Excel file saved to: {excel_path}")
    except:
        print("Could not save Excel file (openpyxl not available)")
    
    print(f"Data saved to: {json_path}, {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Summarize all metrics.txt files')
    parser.add_argument('--input_dir', default='batch_test_results',
                       help='Directory containing metrics files (default: batch_test_results)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for summary (default: same as input_dir)')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Scanning for metrics files in: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 查找所有metrics文件
    metrics_files = find_all_metrics_files(args.input_dir)
    
    if not metrics_files:
        print("No metrics files found!")
        return
    
    # 解析所有metrics文件
    print("Parsing metrics files...")
    all_metrics = []
    failed_files = []
    
    for filepath in metrics_files:
        metrics = parse_metrics_file(filepath)
        if metrics:
            all_metrics.append(metrics)
        else:
            failed_files.append(filepath)
    
    if failed_files:
        print(f"Warning: Failed to parse {len(failed_files)} files:")
        for f in failed_files:
            print(f"  - {f}")
    
    if not all_metrics:
        print("No valid metrics data found!")
        return
    
    print(f"Successfully parsed {len(all_metrics)} metrics files")
    
    # 计算汇总统计
    print("Calculating summary statistics...")
    summary_stats, df = calculate_summary_statistics(all_metrics)
    
    # 生成报告
    print("Generating summary report...")
    report_path = generate_summary_report(all_metrics, summary_stats, df, args.output_dir)
    
    # 保存数据
    print("Saving summary data...")
    save_summary_data(all_metrics, summary_stats, df, args.output_dir)
    
    # 创建可视化（如果请求）
    if args.create_plots:
        print("Creating visualizations...")
        try:
            create_visualizations(df, summary_stats, args.output_dir)
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    # 显示主要结果
    print("\n" + "="*80)
    print("METRICS SUMMARY COMPLETED")
    print("="*80)
    
    if 'dice' in summary_stats:
        dice_stats = summary_stats['dice']
        print(f"📊 Patients analyzed: {len(all_metrics)}")
        print(f"🎯 Average Dice Score: {dice_stats['mean']:.4f} ± {dice_stats['std']:.4f}")
        print(f"📈 Best performance: {dice_stats['max']:.4f}")
        print(f"📉 Worst performance: {dice_stats['min']:.4f}")
    
    print(f"📄 Detailed report: {report_path}")
    print(f"📁 All outputs in: {args.output_dir}")

if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助
    import sys
    if len(sys.argv) == 1:
        print("Metrics Summary Tool")
        print("=" * 30)
        print("\nUsage examples:")
        print("python calc.py")
        print("python calc.py --input_dir batch_test_results")
        print("python calc.py --create_plots")
        print("python calc.py --input_dir results --output_dir summary --create_plots")
        print("\nThis script will:")
        print("- Find all *_metrics.txt files in the directory")
        print("- Parse and summarize all metrics")
        print("- Generate comprehensive summary report")
        print("- Create visualizations (if requested)")
        print("- Export data in multiple formats")
    else:
        main()