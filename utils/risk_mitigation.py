# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import psutil
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import warnings


class TrainingStabilityMonitor:
    """Training Stability Monitor"""
    
    def __init__(self, 
                 window_size: int = 100,
                 gradient_threshold: float = 10.0,
                 loss_variance_threshold: float = 1.0):
        self.window_size = window_size
        self.gradient_threshold = gradient_threshold
        self.loss_variance_threshold = loss_variance_threshold
        
        # History
        self.loss_history = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        
        # Anomaly counters
        self.gradient_explosion_count = 0
        self.nan_loss_count = 0
        self.high_variance_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def update(self, 
               loss: float,
               model: nn.Module,
               learning_rate: float) -> Dict[str, bool]:
        """
        Update monitoring status
        
        Returns:
            Dictionary of anomaly detection results
        """
        alerts = {
            'gradient_explosion': False,
            'nan_loss': False,
            'high_variance': False,
            'training_unstable': False
        }
        
        # Check for NaN loss
        if np.isnan(loss) or np.isinf(loss):
            self.nan_loss_count += 1
            alerts['nan_loss'] = True
            self.logger.warning(f"NaN/Inf loss detected: {loss}")
        
        # Calculate gradient norm
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            # Check for gradient explosion
            if total_norm > self.gradient_threshold:
                self.gradient_explosion_count += 1
                alerts['gradient_explosion'] = True
                self.logger.warning(f"Gradient explosion detected: {total_norm}")
        
        # Update history
        self.loss_history.append(loss)
        self.learning_rates.append(learning_rate)
        
        # Check loss variance
        if len(self.loss_history) >= self.window_size:
            loss_variance = np.var(list(self.loss_history))
            if loss_variance > self.loss_variance_threshold:
                self.high_variance_count += 1
                alerts['high_variance'] = True
                self.logger.warning(f"High loss variance detected: {loss_variance}")
        
        # Comprehensive stability judgment
        recent_alerts = (
            self.gradient_explosion_count > 5 or
            self.nan_loss_count > 3 or
            self.high_variance_count > 10
        )
        
        if recent_alerts:
            alerts['training_unstable'] = True
            self.logger.error("Training instability detected!")
        
        return alerts
    
    def get_statistics(self) -> Dict[str, float]:
        """Get monitoring statistics"""
        stats = {}
        
        if self.loss_history:
            stats['loss_mean'] = np.mean(self.loss_history)
            stats['loss_std'] = np.std(self.loss_history)
            stats['loss_trend'] = self._calculate_trend(list(self.loss_history))
        
        if self.gradient_norms:
            stats['gradient_norm_mean'] = np.mean(self.gradient_norms)
            stats['gradient_norm_max'] = np.max(self.gradient_norms)
        
        stats['gradient_explosion_count'] = self.gradient_explosion_count
        stats['nan_loss_count'] = self.nan_loss_count
        stats['high_variance_count'] = self.high_variance_count
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive value means rising, negative value means falling)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression to calculate the slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope
    
    def reset_counters(self):
        """Reset anomaly counters"""
        self.gradient_explosion_count = 0
        self.nan_loss_count = 0
        self.high_variance_count = 0


class MemoryMonitor:
    """Memory Usage Monitor"""
    
    def __init__(self, 
                 memory_threshold: float = 0.9,  # 90% memory usage threshold
                 check_interval: int = 10):      # Check interval (steps)
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.step_count = 0
        
        self.peak_memory = 0.0
        self.memory_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
    
    def check_memory(self) -> Dict[str, any]:
        """Check memory usage"""
        self.step_count += 1
        
        if self.step_count % self.check_interval != 0:
            return {'memory_ok': True}
        
        # GPU memory check
        gpu_memory_info = {}
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            gpu_usage_ratio = gpu_memory_allocated / gpu_memory_total
            
            gpu_memory_info = {
                'gpu_allocated': gpu_memory_allocated,
                'gpu_reserved': gpu_memory_reserved,
                'gpu_total': gpu_memory_total,
                'gpu_usage_ratio': gpu_usage_ratio
            }
            
            if gpu_usage_ratio > self.memory_threshold:
                self.logger.warning(f"High GPU memory usage: {gpu_usage_ratio:.2%}")
        
        # System memory check
        system_memory = psutil.virtual_memory()
        system_usage_ratio = system_memory.percent / 100
        
        system_memory_info = {
            'system_used': system_memory.used / 1024**3,  # GB
            'system_total': system_memory.total / 1024**3,
            'system_usage_ratio': system_usage_ratio
        }
        
        if system_usage_ratio > self.memory_threshold:
            self.logger.warning(f"High system memory usage: {system_usage_ratio:.2%}")
        
        # Update history
        current_memory = gpu_memory_info.get('gpu_usage_ratio', system_usage_ratio)
        self.memory_history.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        memory_ok = (
            gpu_memory_info.get('gpu_usage_ratio', 0) < self.memory_threshold and
            system_usage_ratio < self.memory_threshold
        )
        
        return {
            'memory_ok': memory_ok,
            'gpu_info': gpu_memory_info,
            'system_info': system_memory_info,
            'peak_memory': self.peak_memory
        }
    
    def cleanup_memory(self):
        """Clean up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        self.logger.info("Memory cleanup performed")


class PerformanceMonitor:
    """Performance Monitor"""
    
    def __init__(self, baseline_metrics: Optional[Dict[str, float]] = None):
        self.baseline_metrics = baseline_metrics or {}
        self.current_metrics = {}
        self.performance_history = deque(maxlen=50)
        
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.current_metrics = metrics.copy()
        self.performance_history.append(metrics)
    
    def check_performance_degradation(self, 
                                    tolerance: float = 0.05) -> Dict[str, bool]:
        """Check for performance degradation"""
        alerts = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in self.current_metrics:
                current_value = self.current_metrics[metric_name]
                degradation = (baseline_value - current_value) / baseline_value
                
                if degradation > tolerance:
                    alerts[f'{metric_name}_degraded'] = True
                    self.logger.warning(
                        f"Performance degradation in {metric_name}: "
                        f"{baseline_value:.4f} -> {current_value:.4f} "
                        f"({degradation:.2%} drop)"
                    )
                else:
                    alerts[f'{metric_name}_degraded'] = False
        
        return alerts
    
    def get_performance_trend(self) -> Dict[str, float]:
        """Get performance trend"""
        if len(self.performance_history) < 5:
            return {}
        
        trends = {}
        recent_metrics = list(self.performance_history)[-5:]
        
        for metric_name in recent_metrics[0].keys():
            values = [m[metric_name] for m in recent_metrics]
            trend = self._calculate_trend(values)
            trends[f'{metric_name}_trend'] = trend
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope


class RiskMitigationManager:
    """Risk Mitigation Manager"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize monitors
        self.stability_monitor = TrainingStabilityMonitor(
            gradient_threshold=config.get('gradient_clip', 1.0) * 2,
            loss_variance_threshold=config.get('loss_variance_threshold', 1.0)
        )
        
        self.memory_monitor = MemoryMonitor(
            memory_threshold=config.get('memory_threshold', 0.9)
        )
        
        self.performance_monitor = PerformanceMonitor()
        
        # Mitigation strategies
        self.mitigation_strategies = {
            'gradient_explosion': self._handle_gradient_explosion,
            'nan_loss': self._handle_nan_loss,
            'high_variance': self._handle_high_variance,
            'memory_overflow': self._handle_memory_overflow,
            'performance_degradation': self._handle_performance_degradation
        }
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_and_mitigate(self,
                           loss: float,
                           model: nn.Module,
                           optimizer: torch.optim.Optimizer,
                           metrics: Dict[str, float]) -> Dict[str, any]:
        """Monitor and execute risk mitigation"""
        mitigation_actions = []
        
        # Training stability monitoring
        stability_alerts = self.stability_monitor.update(
            loss, model, optimizer.param_groups[0]['lr']
        )
        
        # Memory monitoring
        memory_status = self.memory_monitor.check_memory()
        
        # Performance monitoring
        self.performance_monitor.update_metrics(metrics)
        performance_alerts = self.performance_monitor.check_performance_degradation()
        
        # Execute mitigation strategies
        all_alerts = {**stability_alerts, **performance_alerts}
        if not memory_status['memory_ok']:
            all_alerts['memory_overflow'] = True
        
        for alert_type, is_active in all_alerts.items():
            if is_active and alert_type in self.mitigation_strategies:
                action = self.mitigation_strategies[alert_type](
                    model, optimizer, loss, metrics
                )
                mitigation_actions.append(action)
        
        return {
            'alerts': all_alerts,
            'actions': mitigation_actions,
            'stability_stats': self.stability_monitor.get_statistics(),
            'memory_status': memory_status,
            'performance_trends': self.performance_monitor.get_performance_trend()
        }
    
    def _handle_gradient_explosion(self, model, optimizer, loss, metrics):
        """Handle gradient explosion"""
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        self.logger.info("Applied gradient explosion mitigation: LR reduced, gradients clipped")
        return "gradient_explosion_mitigation"
    
    def _handle_nan_loss(self, model, optimizer, loss, metrics):
        """Handle NaN loss"""
        # Reset optimizer state
        optimizer.zero_grad()
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        
        self.logger.warning("Applied NaN loss mitigation: optimizer reset, LR reduced")
        return "nan_loss_mitigation"
    
    def _handle_high_variance(self, model, optimizer, loss, metrics):
        """Handle high variance"""
        # Increase regularization
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] *= 1.5
        
        self.logger.info("Applied high variance mitigation: increased regularization")
        return "high_variance_mitigation"
    
    def _handle_memory_overflow(self, model, optimizer, loss, metrics):
        """Handle memory overflow"""
        # Clean up memory
        self.memory_monitor.cleanup_memory()
        
        # Suggest reducing batch size
        self.logger.warning("Memory overflow detected. Consider reducing batch size.")
        return "memory_overflow_mitigation"
    
    def _handle_performance_degradation(self, model, optimizer, loss, metrics):
        """Handle performance degradation"""
        # Suggest rolling back to a previous checkpoint
        self.logger.warning("Performance degradation detected. Consider loading previous checkpoint.")
        return "performance_degradation_warning"


class AutoRecoverySystem:
    """Auto Recovery System"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        self.checkpoint_dir = checkpoint_dir
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float]):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def attempt_recovery(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer) -> bool:
        """Attempt to recover from a checkpoint"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.logger.error("Maximum recovery attempts reached")
            return False
        
        try:
            # Find the latest checkpoint
            import glob
            import os
            
            checkpoint_files = glob.glob(f"{self.checkpoint_dir}/checkpoint_epoch_*.pt")
            if not checkpoint_files:
                self.logger.error("No checkpoint files found")
                return False
            
            # Sort by epoch and select the latest
            latest_checkpoint = max(checkpoint_files, 
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Load the checkpoint
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.recovery_attempts += 1
            self.logger.info(f"Recovery successful from {latest_checkpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            return False
