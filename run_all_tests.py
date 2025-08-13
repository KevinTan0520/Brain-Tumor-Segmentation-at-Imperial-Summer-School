import os
import subprocess
import sys
import time
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# 获取当前文件所在目录并设置为工作目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"Current working directory set to: {CURRENT_DIR}")

def find_all_patients(dataset_dir):
    """找到所有patient的文件对"""
    patient_pairs = []
    train_dir = os.path.join(dataset_dir, 'train')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        return patient_pairs
    
    print(f"Scanning for patients in: {train_dir}")
    
    for patient_id in sorted(os.listdir(train_dir)):
        patient_dir = os.path.join(train_dir, patient_id)
        if os.path.isdir(patient_dir):
            fla_path = os.path.join(patient_dir, f"{patient_id}_fla.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
            
            if os.path.exists(fla_path) and os.path.exists(seg_path):
                patient_pairs.append({
                    'patient_id': patient_id,
                    'fla_path': fla_path,
                    'seg_path': seg_path
                })
                print(f"Found patient: {patient_id}")
            else:
                print(f"Warning: Incomplete data for patient {patient_id}")
                if not os.path.exists(fla_path):
                    print(f"  Missing: {fla_path}")
                if not os.path.exists(seg_path):
                    print(f"  Missing: {seg_path}")
    
    print(f"Total patients found: {len(patient_pairs)}")
    return patient_pairs

def generate_timestamp_folder_name(base_name="summary"):
    """生成带时间戳的文件夹名称"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

def check_model_compatibility(model_path):
    """检查模型文件的格式和兼容性"""
    try:
        import torch
        
        print(f"Checking model compatibility: {model_path}")
        print(f"PyTorch version: {torch.__version__}")
        
        # 尝试加载模型以检查格式
        try:
            # 首先尝试安全加载
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print("Model can be loaded with weights_only=True (safe)")
            return True, "safe_weights_only"
        except Exception as e1:
            print(f"Safe loading failed: {e1}")
            
            try:
                # 尝试不安全加载
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print("Model can be loaded with weights_only=False")
                
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        print("Found model_state_dict (new training format)")
                        return True, "checkpoint_dict"
                    else:
                        print("Direct state_dict format")
                        return True, "direct_state_dict"
                else:
                    print("Legacy model format")
                    return True, "legacy"
                    
            except Exception as e2:
                print(f"Model loading failed completely: {e2}")
                return False, f"failed: {e2}"
                
    except ImportError:
        print("PyTorch not available for model checking")
        return False, "pytorch_not_available"

def run_single_test(patient_info, model_path, output_base_dir, compare_script_path, model_format=None):
    """运行单个patient的测试"""
    patient_id = patient_info['patient_id']
    fla_path = patient_info['fla_path']
    seg_path = patient_info['seg_path']
    
    # 为每个patient创建独立的输出目录
    patient_output_dir = os.path.join(output_base_dir, f"patient_{patient_id}")
    
    print(f"\n{'='*60}")
    print(f"Testing Patient: {patient_id}")
    print(f"FLA file: {fla_path}")
    print(f"SEG file: {seg_path}")
    print(f"Output dir: {patient_output_dir}")
    if model_format:
        print(f"Model format: {model_format}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        compare_script_path,
        "--input_path", fla_path,
        "--ground_truth_path", seg_path,
        "--model_path", model_path,
        "--output_dir", patient_output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 设置环境变量以便子进程知道模型格式
        env = os.environ.copy()
        if model_format:
            env['MODEL_FORMAT_HINT'] = model_format
        # 设置编码环境变量以避免Unicode错误
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 运行命令，增加超时时间以适应新模型的加载时间
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, encoding='utf-8', errors='ignore')  # 10分钟超时
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"Patient {patient_id} completed successfully in {duration:.2f} seconds")
            return True, duration, result.stdout
        else:
            print(f"Patient {patient_id} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            
            # 检查是否是模型加载错误
            if "Error loading model" in result.stderr or "torch.load" in result.stderr:
                print("This appears to be a model loading issue. Trying alternative approach...")
                
                # 可以在这里添加重试逻辑或者特殊处理
                return False, duration, f"Model loading error: {result.stderr}"
            
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"Patient {patient_id} timed out after 10 minutes")
        return False, 600, "Timeout (10 minutes)"
    except Exception as e:
        print(f"Patient {patient_id} failed with exception: {e}")
        return False, 0, str(e)

def parse_patient_metrics(patient_output_dir, patient_id):
    """解析单个患者的指标文件"""
    metrics_file = os.path.join(patient_output_dir, f"{patient_id}_metrics.txt")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found for patient {patient_id}")
        return None
    
    metrics = {}
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析整体指标
        if "OVERALL METRICS:" in content:
            overall_section = content.split("OVERALL METRICS:")[1].split("\n\n")[0]
            for line in overall_section.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        pass
        
        # 解析统计信息
        if "Total slices:" in content:
            total_slices_match = re.search(r"Total slices:\s+(\d+)", content)
            if total_slices_match:
                metrics['total_slices'] = int(total_slices_match.group(1))
        
        if "Slices with tumor:" in content:
            tumor_slices_match = re.search(r"Slices with tumor:\s+(\d+)", content)
            if tumor_slices_match:
                metrics['slices_with_tumor'] = int(tumor_slices_match.group(1))
        
        if "True tumor volume:" in content:
            true_volume_match = re.search(r"True tumor volume:\s+(\d+)", content)
            if true_volume_match:
                metrics['true_tumor_volume'] = int(true_volume_match.group(1))
        
        if "Pred tumor volume:" in content:
            pred_volume_match = re.search(r"Pred tumor volume:\s+(\d+)", content)
            if pred_volume_match:
                metrics['pred_tumor_volume'] = int(pred_volume_match.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing metrics for patient {patient_id}: {e}")
        return None

def collect_all_metrics(results, output_base_dir):
    """收集所有患者的指标"""
    all_metrics = []
    
    for result in results:
        if result['success']:
            patient_id = result['patient_id']
            patient_output_dir = os.path.join(output_base_dir, f"patient_{patient_id}")
            
            metrics = parse_patient_metrics(patient_output_dir, patient_id)
            if metrics:
                metrics['patient_id'] = patient_id
                metrics['duration'] = result['duration']
                all_metrics.append(metrics)
            else:
                print(f"Could not parse metrics for patient {patient_id}")
    
    return all_metrics

def generate_comprehensive_report(results, all_metrics, output_base_dir, model_info=None):
    """生成全面的测试报告"""
    
    # 基本统计
    total_patients = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_patients - successful
    total_time = sum(r['duration'] for r in results)
    
    # 计算指标统计
    metric_stats = {}
    if all_metrics:
        metric_names = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                metric_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'values': values
                }
    
    # 生成详细报告
    report_file = os.path.join(output_base_dir, "comprehensive_test_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BRAIN TUMOR SEGMENTATION - COMPREHENSIVE TEST REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 添加模型信息
        if model_info:
            f.write(f"Model compatibility: {model_info[1]}\n")
        f.write("\n")
        
        # 执行概要
        f.write("EXECUTION SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total patients processed: {total_patients}\n")
        f.write(f"Successful tests: {successful}\n")
        f.write(f"Failed tests: {failed}\n")
        f.write(f"Success rate: {successful/total_patients*100:.1f}%\n")
        f.write(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per patient: {total_time/total_patients:.2f} seconds\n\n")
        
        if all_metrics:
            # 整体性能统计
            f.write("OVERALL PERFORMANCE STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Number of patients with valid metrics: {len(all_metrics)}\n\n")
            
            # 主要指标统计表格
            f.write("KEY METRICS SUMMARY:\n")
            f.write(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}\n")
            f.write("-"*60 + "\n")
            
            for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score']:
                if metric in metric_stats:
                    stats = metric_stats[metric]
                    f.write(f"{metric.upper():<12} {stats['mean']:.4f}   {stats['std']:.4f}   "
                           f"{stats['min']:.4f}   {stats['max']:.4f}   {stats['median']:.4f}\n")
            
            f.write("\n")
            
            # 分级评估
            f.write("PERFORMANCE GRADING\n")
            f.write("-"*20 + "\n")
            if 'dice' in metric_stats:
                dice_mean = metric_stats['dice']['mean']
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
                
                f.write(f"Overall Model Performance: {grade}\n")
                f.write(f"Average Dice Coefficient: {dice_mean:.4f}\n\n")
            
            # 患者表现排名
            f.write("PATIENT PERFORMANCE RANKING (by Dice Score)\n")
            f.write("-"*50 + "\n")
            
            # 按Dice得分排序
            sorted_patients = sorted(all_metrics, key=lambda x: x.get('dice', 0), reverse=True)
            
            f.write(f"{'Rank':<6} {'Patient':<10} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8}\n")
            f.write("-"*60 + "\n")
            
            for i, patient in enumerate(sorted_patients[:10], 1):  # Top 10
                dice = patient.get('dice', 0)
                iou = patient.get('iou', 0)
                precision = patient.get('precision', 0)
                recall = patient.get('recall', 0)
                f.write(f"{i:<6} {patient['patient_id']:<10} {dice:.4f}   {iou:.4f}   "
                       f"{precision:.4f}     {recall:.4f}\n")
            
            f.write("\n")
            
            # 问题分析
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-"*20 + "\n")
            
            # 找出表现差的患者
            poor_performers = [p for p in all_metrics if p.get('dice', 0) < 0.5]
            if poor_performers:
                f.write(f"Patients with low performance (Dice < 0.5): {len(poor_performers)}\n")
                for p in poor_performers:
                    f.write(f"  - Patient {p['patient_id']}: Dice = {p.get('dice', 0):.4f}\n")
                f.write("\n")
            
            # 体积分析
            total_true_volume = sum(p.get('true_tumor_volume', 0) for p in all_metrics)
            total_pred_volume = sum(p.get('pred_tumor_volume', 0) for p in all_metrics)
            
            if total_true_volume > 0:
                f.write(f"VOLUME ANALYSIS:\n")
                f.write(f"Total true tumor volume: {total_true_volume:,} voxels\n")
                f.write(f"Total predicted volume: {total_pred_volume:,} voxels\n")
                f.write(f"Overall volume ratio: {total_pred_volume/total_true_volume:.4f}\n")
                
                if total_pred_volume > total_true_volume * 1.2:
                    f.write("WARNING: Model tends to OVER-segment (predicts too much tumor)\n")
                elif total_pred_volume < total_true_volume * 0.8:
                    f.write("WARNING: Model tends to UNDER-segment (misses tumor regions)\n")
                else:
                    f.write("SUCCESS: Model shows good volume estimation\n")
                
                f.write("\n")
        
        # 详细患者结果
        f.write("DETAILED PATIENT RESULTS\n")
        f.write("-"*30 + "\n")
        
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            f.write(f"Patient {result['patient_id']}: {status} ({result['duration']:.2f}s)\n")
            
            if result['success'] and all_metrics:
                # 找到对应的指标
                patient_metrics = next((m for m in all_metrics if m['patient_id'] == result['patient_id']), None)
                if patient_metrics:
                    f.write(f"  Dice: {patient_metrics.get('dice', 'N/A'):.4f}, "
                           f"IoU: {patient_metrics.get('iou', 'N/A'):.4f}, "
                           f"Precision: {patient_metrics.get('precision', 'N/A'):.4f}\n")
            elif not result['success']:
                f.write(f"  Error: {result['output'][:100]}...\n")
        
        # 推荐和建议
        f.write("\nRECOMMENDATIONS\n")
        f.write("-"*15 + "\n")
        
        if all_metrics:
            dice_mean = metric_stats.get('dice', {}).get('mean', 0)
            dice_std = metric_stats.get('dice', {}).get('std', 0)
            
            if dice_mean < 0.7:
                f.write("Model performance is below clinical standards. Consider:\n")
                f.write("   - Collecting more training data\n")
                f.write("   - Data augmentation techniques\n")
                f.write("   - Hyperparameter tuning\n")
                f.write("   - Different loss functions (e.g., Focal Loss)\n")
            
            if dice_std > 0.2:
                f.write("High variability in performance across patients. Consider:\n")
                f.write("   - Analyzing patient characteristics\n")
                f.write("   - Stratified training approach\n")
                f.write("   - Ensemble methods\n")
            
            if len(poor_performers) > len(all_metrics) * 0.2:
                f.write("More than 20% of patients show poor performance. Consider:\n")
                f.write("   - Reviewing data quality\n")
                f.write("   - Cross-validation analysis\n")
                f.write("   - Model architecture improvements\n")
        
        if failed > 0:
            f.write(f"{failed} tests failed. Check error logs and:\n")
            f.write("   - Verify data file integrity\n")
            f.write("   - Check system resources\n")
            f.write("   - Review error messages\n")
            f.write("   - Consider model compatibility issues\n")

    # 生成可视化图表
    if all_metrics:
        generate_visualization_plots(all_metrics, metric_stats, output_base_dir)
    
    # 保存指标到JSON
    metrics_json_file = os.path.join(output_base_dir, "all_metrics.json")
    with open(metrics_json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': model_info[1] if model_info else None,
            'summary_stats': metric_stats if all_metrics else {},
            'patient_metrics': all_metrics,
            'execution_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE REPORT GENERATED")
    print(f"{'='*80}")
    print(f"Detailed report: {report_file}")
    print(f"Metrics data: {metrics_json_file}")
    if all_metrics:
        print(f"Visualization plots: {output_base_dir}/visualizations/")
    
    if all_metrics:
        dice_mean = metric_stats.get('dice', {}).get('mean', 0)
        print(f"\nKEY RESULTS:")
        print(f"   Average Dice Score: {dice_mean:.4f}")
        print(f"   Patients tested: {len(all_metrics)}")
        print(f"   Success rate: {successful/total_patients*100:.1f}%")

def generate_visualization_plots(all_metrics, metric_stats, output_base_dir):
    """生成可视化图表"""
    if not all_metrics or not metric_stats:
        return
    
    viz_dir = os.path.join(output_base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置matplotlib参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 主要指标分布直方图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Metrics Distribution', fontsize=16)
    
    metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics_to_plot):
        row, col = i // 3, i % 3
        if metric in metric_stats:
            values = metric_stats[metric]['values']
            axes[row, col].hist(values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].axvline(metric_stats[metric]['mean'], color='red', linestyle='--', 
                                 label=f'Mean: {metric_stats[metric]["mean"]:.3f}')
            axes[row, col].set_title(f'{metric.upper()} Distribution')
            axes[row, col].set_xlabel(metric.upper())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 患者性能排名图
    if 'dice' in metric_stats:
        sorted_patients = sorted(all_metrics, key=lambda x: x.get('dice', 0), reverse=True)
        
        plt.figure(figsize=(12, 8))
        patient_ids = [p['patient_id'] for p in sorted_patients]
        dice_scores = [p.get('dice', 0) for p in sorted_patients]
        
        bars = plt.bar(range(len(patient_ids)), dice_scores, color='lightcoral', alpha=0.7)
        plt.xlabel('Patient ID')
        plt.ylabel('Dice Score')
        plt.title('Patient Performance Ranking (Dice Score)')
        plt.xticks(range(len(patient_ids)), patient_ids, rotation=45)
        
        # 添加平均线
        mean_dice = metric_stats['dice']['mean']
        plt.axhline(mean_dice, color='blue', linestyle='--', label=f'Average: {mean_dice:.3f}')
        
        # 标记表现好和差的患者
        for i, (bar, score) in enumerate(zip(bars, dice_scores)):
            if score >= 0.8:
                bar.set_color('green')
            elif score <= 0.5:
                bar.set_color('red')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'patient_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 指标相关性热图
    try:
        import pandas as pd
        
        df_metrics = pd.DataFrame(all_metrics)
        numeric_cols = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        numeric_cols = [col for col in numeric_cols if col in df_metrics.columns]
        
        if len(numeric_cols) > 1:
            correlation_matrix = df_metrics[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            
            # 添加标签
            plt.xticks(range(len(numeric_cols)), [col.upper() for col in numeric_cols], rotation=45)
            plt.yticks(range(len(numeric_cols)), [col.upper() for col in numeric_cols])
            
            # 添加数值
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
            
            plt.title('Metrics Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("Warning: pandas not available, skipping correlation heatmap")

def generate_summary_report(results, output_base_dir):
    """生成简单总结报告（保持向后兼容）"""
    summary_file = os.path.join(output_base_dir, "test_summary.txt")
    
    total_patients = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_patients - successful
    total_time = sum(r['duration'] for r in results)
    
    print(f"\n{'='*60}")
    print(f"TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients tested: {total_patients}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_patients*100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per patient: {total_time/total_patients:.2f} seconds")
    
    # 写入文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"BRAIN TUMOR SEGMENTATION - BATCH TEST SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"Total patients tested: {total_patients}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {successful/total_patients*100:.1f}%\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per patient: {total_time/total_patients:.2f} seconds\n\n")
        
        f.write(f"DETAILED RESULTS:\n")
        f.write(f"{'-'*60}\n")
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            f.write(f"Patient {result['patient_id']}: {status} ({result['duration']:.2f}s)\n")
            if not result['success']:
                f.write(f"  Error: {result['output'][:100]}...\n")
        
        if failed > 0:
            f.write(f"\nFAILED PATIENTS:\n")
            f.write(f"{'-'*30}\n")
            for result in results:
                if not result['success']:
                    f.write(f"Patient {result['patient_id']}: {result['output']}\n")
    
    print(f"\nSummary report saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Run comparison tests on all patients')
    parser.add_argument('--dataset_dir', default='dataset_segmentation',
                       help='Path to dataset directory (default: dataset_segmentation)')
    parser.add_argument('--model_path', default='best_brain_tumor_model.pth',
                       help='Path to trained model (default: best_brain_tumor_model.pth)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for all results (default: summary_YYYYMMDD_HHMMSS)')
    parser.add_argument('--compare_script', default='compare.py',
                       help='Path to compare script (default: compare.py)')
    parser.add_argument('--patients', nargs='*', default=None,
                       help='Specific patient IDs to test (default: all patients)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip patients that already have results')
    parser.add_argument('--skip_model_check', action='store_true',
                       help='Skip model compatibility check')
    
    args = parser.parse_args()
    
    # 生成带时间戳的输出目录名称
    if args.output_dir is None:
        output_base_dir = os.path.abspath(generate_timestamp_folder_name("summary"))
    else:
        output_base_dir = os.path.abspath(args.output_dir)
    
    # 转换为绝对路径
    dataset_dir = os.path.abspath(args.dataset_dir)
    model_path = os.path.abspath(args.model_path)
    compare_script_path = os.path.abspath(args.compare_script)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_base_dir}")
    print(f"Compare script: {compare_script_path}")
    
    # 检查必要文件
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(compare_script_path):
        print(f"Error: Compare script not found: {compare_script_path}")
        return
    
    # 检查模型兼容性
    model_info = None
    if not args.skip_model_check:
        model_compatible, model_format = check_model_compatibility(model_path)
        model_info = (model_compatible, model_format)
        
        if not model_compatible:
            print(f"Model compatibility check failed: {model_format}")
            print("You can continue with --skip_model_check, but tests may fail")
            choice = input("Continue anyway? (y/N): ").lower()
            if choice != 'y':
                return
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Created output directory: {output_base_dir}")
    
    # 找到所有patients
    all_patients = find_all_patients(dataset_dir)
    
    if not all_patients:
        print("No patients found!")
        return
    
    # 过滤特定patients（如果指定）
    if args.patients:
        all_patients = [p for p in all_patients if p['patient_id'] in args.patients]
        print(f"Testing only specified patients: {args.patients}")
    
    # 跳过已存在的结果（如果指定）
    if args.skip_existing:
        filtered_patients = []
        for patient in all_patients:
            patient_output_dir = os.path.join(output_base_dir, f"patient_{patient['patient_id']}")
            if os.path.exists(patient_output_dir) and os.listdir(patient_output_dir):
                print(f"Skipping patient {patient['patient_id']} (results already exist)")
            else:
                filtered_patients.append(patient)
        all_patients = filtered_patients
    
    if not all_patients:
        print("No patients to test!")
        return
    
    print(f"\nStarting batch testing of {len(all_patients)} patients...")
    
    # 运行测试
    results = []
    
    for i, patient_info in enumerate(all_patients):
        print(f"\nProgress: {i+1}/{len(all_patients)}")
        
        success, duration, output = run_single_test(
            patient_info, model_path, output_base_dir, compare_script_path,
            model_format=model_info[1] if model_info else None
        )
        
        results.append({
            'patient_id': patient_info['patient_id'],
            'success': success,
            'duration': duration,
            'output': output
        })
        
        # 每完成5个patient显示一次进度
        if (i + 1) % 5 == 0:
            successful_so_far = sum(1 for r in results if r['success'])
            print(f"\nProgress update: {i+1}/{len(all_patients)} completed, {successful_so_far} successful")
    
    # 收集所有指标
    print("\nCollecting and analyzing metrics...")
    all_metrics = collect_all_metrics(results, output_base_dir)
    
    # 生成简单总结报告（向后兼容）
    generate_summary_report(results, output_base_dir)
    
    # 生成全面报告
    generate_comprehensive_report(results, all_metrics, output_base_dir, model_info)
    
    print(f"\nBatch testing completed!")
    print(f"Results saved in: {output_base_dir}")

if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    import sys
    if len(sys.argv) == 1:
        print("Brain Tumor Segmentation - Batch Testing Script")
        print("=" * 50)
        print("\nUsage examples:")
        print("python run_all_tests.py")
        print("python run_all_tests.py --patients 001 002 003")
        print("python run_all_tests.py --skip_existing")
        print("python run_all_tests.py --output_dir my_custom_results")
        print("python run_all_tests.py --skip_model_check")
        print("\nDefault output folder naming:")
        example_name = generate_timestamp_folder_name("summary")
        print(f"- Without --output_dir: {example_name}")
        print("- With --output_dir: uses your specified name")
        print("\nThis script will:")
        print("- Check model compatibility with new training formats")
        print("- Find all patients in the dataset")
        print("- Run compare.py for each patient")
        print("- Generate individual result folders")
        print("- Create comprehensive analysis reports with visualizations")
        print("- Provide performance statistics and recommendations")
        print(f"\nCurrent directory: {CURRENT_DIR}")
    else:
        main()