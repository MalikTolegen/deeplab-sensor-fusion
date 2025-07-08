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
        # Define experiment grid with specified batch sizes and sparsities
        batch_sizes = [8, 12, 16]
        sparsities = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        
        # Create pruning configurations (include both pruned and unpruned cases)
        pruning_configs = [
            {'enabled': True, 'target_sparsity': sparsity}
            for sparsity in sparsities[1:]  # Skip 0.0 for pruned configs
        ]
        # Add unpruned case
        pruning_configs.append({'enabled': False})
        
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
                
                # Update pruning config
                for key, value in prune_cfg.items():
                    if hasattr(PRUNING_CFG, key):
                        setattr(PRUNING_CFG, key, value)
                
                # Save experiment config
                config = {
                    'batch_size': batch_size,
                    'pruning_config': prune_cfg,
                    'start_time': datetime.now().isoformat()
                }
                
                with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                try:
                    # Clear CUDA cache before starting
                    import torch
                    torch.cuda.empty_cache()
                    
                    # Record memory before training
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        start_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                    
                    import time
                    start_time = time.time()
                    
                    # Run training
                    best_iou = train()
                    
                    # Calculate training duration
                    training_time = time.time() - start_time
                    
                    # Record memory after training
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        end_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    
                    # Record results
                    result = {
                        'experiment': exp_name,
                        'batch_size': batch_size,
                        'pruning_enabled': prune_cfg['enabled'],
                        'target_sparsity': prune_cfg.get('target_sparsity', 0),
                        'best_iou': float(best_iou) if best_iou is not None else None,
                        'training_time_seconds': training_time,
                        'status': 'completed'
                    }
                    
                    if torch.cuda.is_available():
                        result.update({
                            'initial_memory_mb': start_mem,
                            'final_memory_mb': end_mem,
                            'peak_memory_mb': peak_mem,
                            'memory_usage_mb': end_mem - start_mem
                        })
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error in experiment {exp_name}: {error_msg}")
                    
                    # Add CUDA memory info if available
                    cuda_info = {}
                    if torch.cuda.is_available():
                        try:
                            cuda_info = {
                                'cuda_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                                'cuda_memory_cached_mb': torch.cuda.memory_reserved() / 1024**2,
                                'cuda_max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                                'cuda_device_count': torch.cuda.device_count(),
                                'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                            }
                        except Exception as mem_err:
                            cuda_info = {'cuda_error': str(mem_err)}
                    
                    result = {
                        'experiment': exp_name,
                        'batch_size': batch_size,
                        'pruning_enabled': prune_cfg['enabled'],
                        'target_sparsity': prune_cfg.get('target_sparsity', 0),
                        'error': error_msg,
                        'status': 'failed',
                        'cuda_info': cuda_info,
                        'timestamp': datetime.now().isoformat()
                    }
                
                results.append(result)
                
                # Save results after each experiment with backup
                results_file = os.path.join(self.results_dir, 'all_results.json')
                # Create a backup if file exists
                if os.path.exists(results_file):
                    import shutil
                    backup_file = os.path.join(self.results_dir, 'all_results_backup.json')
                    shutil.copy2(results_file, backup_file)
                
                # Save with pretty printing
                with open(results_file, 'w') as f:
                    # Convert all tensors to python native types for JSON serialization
                    def convert_tensors(obj):
                        if isinstance(obj, dict):
                            return {k: convert_tensors(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_tensors(x) for x in obj]
                        elif hasattr(obj, 'item'):  # For torch.Tensor and numpy arrays
                            return obj.item()
                        return obj
                    
                    json.dump(convert_tensors(results), f, indent=2, sort_keys=True)
                
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
