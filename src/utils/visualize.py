import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict

def plot_metrics(log_dir: str, metrics: List[str] = None):
    """
    Plot training metrics from JSON log files.
    
    Args:
        log_dir: Directory containing the log files
        metrics: List of metric names to plot. If None, plot all metrics.
    """
    # Find all JSON files in the log directory
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    
    for log_file in log_files:
        with open(os.path.join(log_dir, log_file), 'r') as f:
            logs = json.load(f)
        
        # If no specific metrics are requested, plot all metrics
        if metrics is None:
            metrics = list(logs[0].keys())
        
        # Create a figure for each metric
        for metric in metrics:
            if metric in logs[0]:
                plt.figure(figsize=(10, 6))
                values = [log[metric] for log in logs if metric in log]
                plt.plot(range(len(values)), values, label=metric)
                plt.title(f'{metric} over time')
                plt.xlabel('Step')
                plt.ylabel(metric)
                plt.grid(True)
                plt.legend()
                
                # Save the plot
                plt.savefig(os.path.join(log_dir, f'{metric}_plot.png'))
                plt.close()

def plot_comparison(results: Dict[str, List[float]], title: str, xlabel: str, ylabel: str, save_path: str):
    """
    Plot comparison between different models or configurations.
    
    Args:
        results: Dictionary with model names as keys and lists of values as values
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, values in results.items():
        plt.plot(range(len(values)), values, label=model_name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close() 