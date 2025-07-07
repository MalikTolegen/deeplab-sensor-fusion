import os
import json
import copy
from pathlib import Path
from datetime import datetime

from config.model_cfg import *
from main import train, valid
from utils.metrics import calculate_metrics, analyze_pruning

class ExperimentRunner:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_experiments(self):
        # Define experiment grid with larger batch sizes
        batch_sizes = [8, 16, 32]  # Increased minimum batch size
        pruning_configs = [
            {'enabled': True, 'target_sparsity': 0.3, 'grad_accum_steps': 2},
            {'enabled': True, 'target_sparsity': 0.5, 'grad_accum_steps': 2},
            {'enabled': False, 'grad_accum_steps': 1}
        ]
        
        results = []
        
        for batch_size in batch_sizes:
            for prune_cfg in pruning_configs:
                # Create experiment directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_name = f"bs{batch_size}_prune{prune_cfg['enabled']}_sp{prune_cfg.get('target_sparsity', 0)}"
                exp_dir = os.path.join(self.results_dir, f"{timestamp}_{exp_name}")
                os.makedirs(exp_dir, exist_ok=True)
                
                print(f"\n{'='*50}")
                print(f"Running experiment: {exp_name}")
                print(f"{'='*50}")
                
                # Update configs
                TRAIN_DATALOADER_CFG.batch_size = batch_size
                VALID_DATALOADER_CFG.batch_size = batch_size * 2
                
                # Update pruning config and training settings
                for key, value in prune_cfg.items():
                    if hasattr(PRUNING_CFG, key):
                        setattr(PRUNING_CFG, key, value)
                
                # Set gradient accumulation steps
                TRAIN_CFG.grad_accum_steps = prune_cfg.get('grad_accum_steps', 1)
                
                # Save experiment config
                config = {
                    'batch_size': batch_size,
                    'pruning_config': prune_cfg,
                    'start_time': datetime.now().isoformat()
                }
                
                with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                try:
                    # Run training
                    best_iou = train()
                    
                    # Record results
                    result = {
                        'experiment': exp_name,
                        'batch_size': batch_size,
                        'pruning_enabled': prune_cfg['enabled'],
                        'target_sparsity': prune_cfg.get('target_sparsity', 0),
                        'best_iou': best_iou,
                        'status': 'completed'
                    }
                    
                except Exception as e:
                    print(f"Error in experiment {exp_name}: {str(e)}")
                    result = {
                        'experiment': exp_name,
                        'batch_size': batch_size,
                        'pruning_enabled': prune_cfg['enabled'],
                        'target_sparsity': prune_cfg.get('target_sparsity', 0),
                        'error': str(e),
                        'status': 'failed'
                    }
                
                results.append(result)
                
                # Save results after each experiment
                with open(os.path.join(self.results_dir, 'all_results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\nCompleted experiment: {exp_name}")
                print(f"Best IoU: {result.get('best_iou', 'N/A')}")
                print(f"Status: {result['status']}")
                
        return results

    def analyze_results(self):
        """Analyze and visualize experiment results."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        results_file = os.path.join(self.results_dir, 'all_results.json')
        if not os.path.exists(results_file):
            print("No results found. Run experiments first.")
            return
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        df = pd.DataFrame(results)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # Plot IoU by batch size and pruning config
        plt.figure(figsize=(14, 6))
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
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'))
        
        # Save summary
        summary = df.groupby(['batch_size', 'pruning_enabled'])['best_iou'].agg(['mean', 'std'])
        summary.to_csv(os.path.join(self.results_dir, 'summary.csv'))
        
        return df

if __name__ == "__main__":
    from config.model_cfg import *
    
    # Reduce epochs for testing
    TRAIN_CFG.num_epoch = 2  # Set to a small number for testing
    
    runner = ExperimentRunner({
        'batch_sizes': [8],  # Test with one batch size first
        'pruning_configs': [
            {'enabled': False},
            {'enabled': True, 'target_sparsity': 0.3}
        ]
    })
    
    # Run experiments
    results = runner.run_experiments()
    
    # Analyze results
    if any(r['status'] == 'completed' for r in results):
        runner.analyze_results()
        print("\nExperiment complete! Check the 'experiment_results' directory for results.")
    else:
        print("No experiments completed successfully.")
