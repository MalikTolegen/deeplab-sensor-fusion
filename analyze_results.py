import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_dir="experiment_results"):
    """Load all experiment results."""
    results_file = os.path.join(results_dir, 'all_results.json')
    if not os.path.exists(results_file):
        print(f"No results found in {results_file}")
        return None
        
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return pd.DataFrame(results)

def plot_metrics(df, save_dir="experiment_results"):
    """Plot various metrics from experiment results."""
    if df is None or df.empty:
        print("No data to plot")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. IoU by batch size and pruning
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='batch_size',
        y='best_iou',
        hue='pruning_enabled',
        data=df,
        palette='viridis'
    )
    plt.title('Model Performance by Batch Size and Pruning')
    plt.xlabel('Batch Size')
    plt.ylabel('Best Validation IoU')
    plt.legend(title='Pruning Enabled')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_comparison.png'))
    plt.close()
    
    # 2. Training time comparison
    if 'training_time' in df.columns:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x='batch_size',
            y='training_time',
            hue='pruning_enabled',
            data=df,
            palette='viridis'
        )
        plt.title('Training Time by Batch Size and Pruning')
        plt.xlabel('Batch Size')
        plt.ylabel('Training Time (seconds)')
        plt.legend(title='Pruning Enabled')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_time.png'))
        plt.close()
    
    # 3. Model size comparison (if pruning data available)
    if 'model_size_mb' in df.columns:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x='batch_size',
            y='model_size_mb',
            hue='pruning_enabled',
            data=df,
            palette='viridis'
        )
        plt.title('Model Size by Batch Size and Pruning')
        plt.xlabel('Batch Size')
        plt.ylabel('Model Size (MB)')
        plt.legend(title='Pruning Enabled')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_size.png'))
        plt.close()

def generate_report(df, save_dir="experiment_results"):
    """Generate a markdown report with key findings."""
    if df is None or df.empty:
        return
        
    report = "# Experiment Results Report\n\n"
    
    # Best performing configuration
    best_run = df.loc[df['best_iou'].idxmax()]
    report += f"## Best Configuration\n"
    report += f"- **Batch Size**: {best_run['batch_size']}\n"
    report += f"- **Pruning Enabled**: {best_run['pruning_enabled']}\n"
    if best_run['pruning_enabled']:
        report += f"- **Target Sparsity**: {best_run['target_sparsity']*100}%\n"
    report += f"- **Best IoU**: {best_run['best_iou']:.4f}\n\n"
    
    # Summary statistics
    report += "## Summary Statistics\n"
    # Group by batch size and pruning
    grouped = df.groupby(['batch_size', 'pruning_enabled'])
    stats = grouped['best_iou'].agg(['mean', 'std', 'count']).reset_index()
    report += stats.to_markdown(index=False) + "\n\n"
    
    # Save report
    with open(os.path.join(save_dir, 'report.md'), 'w') as f:
        f.write(report)
    
    # Save detailed results as CSV
    df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)

if __name__ == "__main__":
    # Load results
    df = load_results()
    
    if df is not None and not df.empty:
        # Plot metrics
        plot_metrics(df)
        
        # Generate report
        generate_report(df)
        
        print("Analysis complete! Check the 'experiment_results' directory for results.")
        print(f"Best IoU: {df['best_iou'].max():.4f}")
    else:
        print("No valid results to analyze.")
