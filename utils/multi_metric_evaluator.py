# coding: utf-8

"""
Multi-metric tie-breaking evaluation module
Supports both multi-metric priority evaluation and weighted evaluation modes
"""

from typing import Dict, List, Optional, Tuple, Union
from logging import getLogger


class MultiMetricEvaluator:
    """
    Multi-metric evaluator, supports tie-breaking and weighted evaluation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-metric evaluator
        
        Args:
            config: Configuration dictionary, containing multi_metric_evaluation configuration
        """
        self.logger = getLogger()
        
        # Multi-metric evaluation configuration
        self.multi_config = config.get('multi_metric_evaluation', {})
        self.enable = self.multi_config.get('enable', False)
        
        if not self.enable:
            # Backward compatibility mode: use original single-metric evaluation
            self.primary_metric = config.get('valid_metric', 'Recall@20').lower()
            self.valid_metric_bigger = config.get('valid_metric_bigger', True)
            self.logger.info("Using single-metric evaluation mode (backward compatibility)")
            return
            
        # Multi-metric evaluation configuration
        self.primary_metric = self.multi_config.get('primary_metric', 'Recall@20').lower()
        self.tie_breaking_metrics = [m.lower() for m in self.multi_config.get('tie_breaking_metrics', [])]
        self.metric_weights = {k.lower(): v for k, v in self.multi_config.get('metric_weights', {}).items()}
        self.tie_threshold = self.multi_config.get('tie_threshold', 1e-6)
        self.weighted_score_mode = self.multi_config.get('weighted_score_mode', False)
        
        # Determine metric direction (bigger is better vs. smaller is better)
        smaller_metrics = ['rmse', 'mae', 'logloss']
        self.metric_directions = {}
        
        all_metrics = [self.primary_metric] + self.tie_breaking_metrics
        for metric in all_metrics:
            metric_name = metric.split('@')[0]
            self.metric_directions[metric] = False if metric_name in smaller_metrics else True
            
        self.logger.info(f"Enabled multi-metric evaluation mode")
        self.logger.info(f"   Primary metric: {self.primary_metric}")
        self.logger.info(f"   Tie-breaking metrics: {self.tie_breaking_metrics}")
        self.logger.info(f"   Weighted mode: {self.weighted_score_mode}")
        
    def is_better(self, current_result: Dict[str, float], 
                  best_result: Dict[str, float]) -> bool:
        """
        Determine if the current result is better than the best result
        
        Args:
            current_result: Current evaluation result
            best_result: Historical best result
            
        Returns:
            bool: Whether it is better
        """
        if not self.enable:
            # Single-metric mode
            current_score = current_result.get(self.primary_metric, 0.0)
            best_score = best_result.get(self.primary_metric, -1.0)
            
            if self.valid_metric_bigger:
                return current_score > best_score
            else:
                return current_score < best_score
                
        if self.weighted_score_mode:
            return self._weighted_comparison(current_result, best_result)
        else:
            return self._tie_breaking_comparison(current_result, best_result)
    
    def _weighted_comparison(self, current_result: Dict[str, float], 
                           best_result: Dict[str, float]) -> bool:
        """
        Weighted evaluation mode: calculate weighted scores for comparison
        """
        current_score = self._calculate_weighted_score(current_result)
        best_score = self._calculate_weighted_score(best_result)
        
        self.logger.debug(f"Weighted score comparison: current={current_score:.6f}, best={best_score:.6f}")
        return current_score > best_score
    
    def _calculate_weighted_score(self, result: Dict[str, float]) -> float:
        """
        Calculate weighted score
        """
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.metric_weights.items():
            if metric in result:
                score = result[metric]
                # If it is a smaller-is-better metric, convert it to bigger-is-better
                if not self.metric_directions.get(metric, True):
                    score = -score
                    
                weighted_score += score * weight
                total_weight += weight
                
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _tie_breaking_comparison(self, current_result: Dict[str, float], 
                               best_result: Dict[str, float]) -> bool:
        """
        Tie-breaking evaluation mode: compare level by level according to priority
        """
        # 1. First, compare the primary metric
        current_primary = current_result.get(self.primary_metric, 0.0)
        best_primary = best_result.get(self.primary_metric, -1.0)
        
        primary_better = self._compare_single_metric(
            current_primary, best_primary, self.primary_metric
        )
        
        # If there is a significant difference in the primary metric, return the result directly
        if abs(current_primary - best_primary) > self.tie_threshold:
            self.logger.debug(f"Primary metric decides: {self.primary_metric} "
                            f"current={current_primary:.6f}, best={best_primary:.6f}")
            return primary_better
            
        # 2. Primary metric is a tie, use tie-breaking metrics
        self.logger.debug(f"Primary metric tie ({self.primary_metric}={current_primary:.6f}), "
                         f"using tie-breaking metrics")
        
        for tie_metric in self.tie_breaking_metrics:
            if tie_metric not in current_result or tie_metric not in best_result:
                continue
                
            current_tie = current_result[tie_metric]
            best_tie = best_result[tie_metric]
            
            # If there is also a significant difference in the tie-breaking metric
            if abs(current_tie - best_tie) > self.tie_threshold:
                tie_better = self._compare_single_metric(
                    current_tie, best_tie, tie_metric
                )
                self.logger.debug(f"Tie-breaking metric decides: {tie_metric} "
                                f"current={current_tie:.6f}, best={best_tie:.6f}")
                return tie_better
                
        # 3. All metrics are ties, keep the current best
        self.logger.debug("All metrics are ties, keeping the current best")
        return False
    
    def _compare_single_metric(self, current: float, best: float, metric: str) -> bool:
        """
        Compare a single metric
        """
        is_bigger_better = self.metric_directions.get(metric, True)
        
        if is_bigger_better:
            return current > best
        else:
            return current < best
    
    def get_evaluation_score(self, result: Dict[str, float]) -> float:
        """
        Get the evaluation score for early stopping
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            float: Evaluation score
        """
        if not self.enable:
            return result.get(self.primary_metric, 0.0)
            
        if self.weighted_score_mode:
            return self._calculate_weighted_score(result)
        else:
            return result.get(self.primary_metric, 0.0)
    
    def format_result_summary(self, result: Dict[str, float]) -> str:
        """
        Format the result summary, highlighting key metrics
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            str: Formatted result string
        """
        if not self.enable:
            # Single-metric mode
            primary_score = result.get(self.primary_metric, 0.0)
            return f"{self.primary_metric}: {primary_score:.4f}"
            
        # Multi-metric mode
        summary_parts = []
        
        # Primary metric
        if self.primary_metric in result:
            summary_parts.append(f"🎯{self.primary_metric}: {result[self.primary_metric]:.4f}")
            
        # Tie-breaking metrics
        for metric in self.tie_breaking_metrics:
            if metric in result:
                summary_parts.append(f"{metric}: {result[metric]:.4f}")
                
        # Weighted score
        if self.weighted_score_mode:
            weighted_score = self._calculate_weighted_score(result)
            summary_parts.append(f"⚖️Weighted Score: {weighted_score:.4f}")
            
        return " | ".join(summary_parts)


def enhanced_early_stopping(current_result: Dict[str, float], 
                           best_result: Dict[str, float],
                           evaluator: MultiMetricEvaluator,
                           cur_step: int, 
                           max_step: int) -> Tuple[Dict[str, float], int, bool, bool]:
    """
    Enhanced early stopping function, supports multi-metric evaluation
    
    Args:
        current_result: Current evaluation result
        best_result: Historical best result
        evaluator: Multi-metric evaluator
        cur_step: Current number of consecutive unimproved steps
        max_step: Maximum allowed number of unimproved steps
        
    Returns:
        tuple: (best result, number of consecutive unimproved steps, whether to stop, whether to update)
    """
    stop_flag = False
    update_flag = False
    
    # Use the multi-metric evaluator to determine if it is better
    if evaluator.is_better(current_result, best_result):
        cur_step = 0
        best_result = current_result.copy()
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
            
    return best_result, cur_step, stop_flag, update_flag
