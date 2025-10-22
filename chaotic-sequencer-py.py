import json
import math
import os
import random
import secrets
import statistics
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StepType(Enum):
    INITIAL = "initial"
    RANDOM_WALK = "random_walk"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE_NOISE = "additive_noise"

@dataclass
class ChaoticConfig:
    """Configuration for chaotic sequence generation with validation"""
    volatility: float = 0.7
    trend_strength: float = 0.3
    mean_reversion: float = 0.2
    min_value: int = 1
    max_value: int = 1000
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0 <= self.volatility <= 1:
            raise ValueError("Volatility must be between 0 and 1")
        if not 0 <= self.trend_strength <= 1:
            raise ValueError("Trend strength must be between 0 and 1")
        if not 0 <= self.mean_reversion <= 1:
            raise ValueError("Mean reversion must be between 0 and 1")
        if self.min_value >= self.max_value:
            raise ValueError("Min value must be less than max value")
    
    @classmethod
    def default_config(cls) -> 'ChaoticConfig':
        """Returns a sensible default configuration"""
        return cls(
            volatility=0.7,
            trend_strength=0.3,
            mean_reversion=0.2,
            min_value=1,
            max_value=1000
        )

class SecureRandom:
    """Wrapper for secure random number generation with caching and fallbacks"""
    
    @staticmethod
    def int(n: int) -> int:
        """Generate cryptographically secure random integer in [0, n)"""
        if n <= 0:
            return 0
        try:
            return secrets.randbelow(n)
        except (ValueError, NotImplementedError):
            logger.warning("Falling back to system random for integer generation")
            return random.randint(0, n - 1)
    
    @staticmethod
    def float() -> float:
        """Generate cryptographically secure random float in [0, 1)"""
        try:
            random_bytes = secrets.token_bytes(8)
            random_int = int.from_bytes(random_bytes, byteorder='big')
            return (random_int % (1 << 53)) / (1 << 53)
        except (ValueError, NotImplementedError):
            logger.warning("Falling back to system random for float generation")
            return random.random()
    
    @staticmethod
    def choice(seq: List[Any]) -> Any:
        """Choose a random element from sequence"""
        if not seq:
            raise ValueError("Sequence cannot be empty")
        return seq[SecureRandom.int(len(seq))]

class GenerationStrategy(Protocol):
    """Protocol for generation strategies"""
    def generate_next(self, prev1: int, prev2: int, running_mean: float, 
                     chaos_factor: float, config: ChaoticConfig) -> int: ...

class TrendFollowingStrategy:
    def generate_next(self, prev1: int, prev2: int, running_mean: float,
                     chaos_factor: float, config: ChaoticConfig) -> int:
        trend = prev1 - prev2
        return prev1 + int(float(trend) * config.trend_strength) + int(chaos_factor * prev1 * 0.5)

class MeanReversionStrategy:
    def generate_next(self, prev1: int, prev2: int, running_mean: float,
                     chaos_factor: float, config: ChaoticConfig) -> int:
        deviation = float(prev1) - running_mean
        return prev1 - int(deviation * config.mean_reversion) + int(chaos_factor * prev1 * 0.3)

class MultiplicativeStrategy:
    def generate_next(self, prev1: int, prev2: int, running_mean: float,
                     chaos_factor: float, config: ChaoticConfig) -> int:
        factors = [0.3, 0.7, 1.3, 1.7, 2.0, -0.5]
        factor = SecureRandom.choice(factors)
        return int(float(prev1) * factor) + int(chaos_factor * 10)

class AdditiveNoiseStrategy:
    def generate_next(self, prev1: int, prev2: int, running_mean: float,
                     chaos_factor: float, config: ChaoticConfig) -> int:
        noise = SecureRandom.int(21) - 10
        return prev1 + (prev1 - prev2) // 2 + noise

def clamp(value: int, min_val: int, max_val: int) -> int:
    """Ensures value stays within min-max range"""
    return max(min_val, min(max_val, value))

class ChaoticSequenceGenerator:
    """Main generator using strategy pattern for different generation methods"""
    
    def __init__(self, config: ChaoticConfig):
        self.config = config
        self.strategies = {
            StepType.TREND_FOLLOWING: TrendFollowingStrategy(),
            StepType.MEAN_REVERSION: MeanReversionStrategy(),
            StepType.MULTIPLICATIVE: MultiplicativeStrategy(),
            StepType.ADDITIVE_NOISE: AdditiveNoiseStrategy()
        }
    
    def generate_sequence(self, n: int) -> List[Dict[str, Any]]:
        """Generates a chaotic transaction sequence of n steps"""
        if n <= 0:
            raise ValueError("The number of steps must be a positive integer")
        if n < 2:
            raise ValueError("Sequence length must be at least 2 for proper chaotic behavior")

        sequence = self._initialize_sequence(n)
        return self._generate_full_sequence(sequence, n)
    
    def _initialize_sequence(self, n: int) -> List[int]:
        """Initialize the first two sequence values"""
        sequence = [0] * n
        sequence[0] = SecureRandom.int(self.config.max_value - self.config.min_value + 1) + self.config.min_value
        sequence[1] = clamp(
            sequence[0] + SecureRandom.int(21) - 10,
            self.config.min_value,
            self.config.max_value
        )
        return sequence
    
    def _generate_full_sequence(self, sequence: List[int], n: int) -> List[Dict[str, Any]]:
        """Generate the full sequence using various strategies"""
        log = [
            {"step": 0, "value": sequence[0], "type": StepType.INITIAL.value},
            {"step": 1, "value": sequence[1], "type": StepType.RANDOM_WALK.value}
        ]
        
        running_mean = (sequence[0] + sequence[1]) / 2.0
        
        for i in range(2, n):
            strategy_type = self._select_strategy()
            strategy = self.strategies[strategy_type]
            
            next_value = self._apply_strategy(strategy, sequence, i, running_mean)
            next_value = self._apply_volatility(next_value)
            next_value = clamp(next_value, self.config.min_value, self.config.max_value)
            
            sequence[i] = next_value
            running_mean = self._update_running_mean(running_mean, next_value, i)
            
            log.append({"step": i, "value": next_value, "type": strategy_type.value})
        
        return log
    
    def _select_strategy(self) -> StepType:
        """Select a generation strategy based on random weights"""
        rand_val = SecureRandom.float()
        if rand_val < 0.25:
            return StepType.TREND_FOLLOWING
        elif rand_val < 0.5:
            return StepType.MEAN_REVERSION
        elif rand_val < 0.75:
            return StepType.MULTIPLICATIVE
        else:
            return StepType.ADDITIVE_NOISE
    
    def _apply_strategy(self, strategy: GenerationStrategy, sequence: List[int], 
                       i: int, running_mean: float) -> int:
        """Apply the selected strategy to generate next value"""
        chaos_factor = SecureRandom.float() * 2 - 1
        return strategy.generate_next(sequence[i-1], sequence[i-2], running_mean, chaos_factor, self.config)
    
    def _apply_volatility(self, value: int) -> int:
        """Apply volatility effect to the value"""
        chaos_factor = SecureRandom.float() * 2 - 1
        volatility_effect = int(chaos_factor * value * self.config.volatility)
        return value + volatility_effect
    
    def _update_running_mean(self, running_mean: float, new_value: int, index: int) -> float:
        """Update the running mean with new value"""
        return (running_mean * index + new_value) / (index + 1)

def calculate_quantile(values: List[int], quantile: float) -> float:
    """Computes the specified quantile (0.0 to 1.0)"""
    if not values:
        return 0.0
        
    sorted_vals = sorted(values)
    pos = quantile * (len(sorted_vals) - 1)
    lower = int(pos)
    upper = lower + 1
    weight = pos - lower

    if upper >= len(sorted_vals):
        return float(sorted_vals[lower])
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)

def calculate_trend_strength(values: List[int]) -> float:
    """Measures how trending the sequence is"""
    if len(values) < 2:
        return 0.0

    up, down = 0, 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            up += 1
        elif values[i] < values[i - 1]:
            down += 1

    total = up + down
    if total == 0:
        return 0.0
    return abs(up - down) / total

def calculate_volatility(values: List[int]) -> float:
    """Measures the sequence volatility"""
    if len(values) < 2:
        return 0.0

    total_change = 0.0
    for i in range(1, len(values)):
        change = abs(float(values[i]) - float(values[i - 1]))
        total_change += change
        
    return total_change / (len(values) - 1)

class AdvancedStatistics:
    """Advanced statistical analysis with NumPy integration"""
    
    @staticmethod
    def calculate_advanced_stats(values: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the sequence"""
        if not values:
            raise ValueError("Cannot calculate statistics for empty sequence")
        
        stats = {}
        
        # Use NumPy for performance if available, otherwise fallback
        if HAS_NUMPY and len(values) > 1:
            arr = np.array(values)
            stats.update({
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "stdev": float(np.std(arr, ddof=1)),  # Sample standard deviation
                "variance": float(np.var(arr, ddof=1)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "q1": float(np.percentile(arr, 25)),
                "q3": float(np.percentile(arr, 75)),
                "skewness": float(AdvancedStatistics._skewness(arr)),
                "kurtosis": float(AdvancedStatistics._kurtosis(arr)),
            })
        else:
            # Pure Python implementation
            stats.update(AdvancedStatistics._calculate_basic_stats_python(values))
        
        # Additional statistics
        stats.update({
            "count": len(values),
            "trend_strength": calculate_trend_strength(values),
            "volatility": calculate_volatility(values),
            "entropy": AdvancedStatistics._calculate_entropy(values),
            "autocorrelation": AdvancedStatistics._calculate_autocorrelation(values),
            "iqr": stats["q3"] - stats["q1"],
            "coefficient_of_variation": stats["stdev"] / stats["mean"] if stats["mean"] != 0 else 0.0,
        })
        
        return stats
    
    @staticmethod
    def _calculate_basic_stats_python(values: List[int]) -> Dict[str, Any]:
        """Calculate basic statistics using pure Python"""
        stats = {}
        
        sorted_vals = sorted(values)
        stats["min"] = sorted_vals[0]
        stats["max"] = sorted_vals[-1]
        stats["mean"] = statistics.mean(values) if len(values) > 1 else float(values[0])
        stats["median"] = statistics.median(values)
        stats["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        stats["variance"] = stats["stdev"] ** 2
        stats["q1"] = calculate_quantile(values, 0.25)
        stats["q3"] = calculate_quantile(values, 0.75)
        stats["skewness"] = 0.0  # Simplified
        stats["kurtosis"] = 0.0  # Simplified
        
        return stats
    
    @staticmethod
    def _skewness(arr) -> float:
        """Calculate skewness of the distribution"""
        if HAS_NUMPY and len(arr) > 2:
            try:
                from scipy.stats import skew
                return skew(arr)
            except ImportError:
                pass
        # Simplified skewness calculation
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))
    
    @staticmethod
    def _kurtosis(arr) -> float:
        """Calculate kurtosis of the distribution"""
        if HAS_NUMPY and len(arr) > 3:
            try:
                from scipy.stats import kurtosis
                return kurtosis(arr)
            except ImportError:
                pass
        # Simplified kurtosis calculation
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4)) - 3
    
    @staticmethod
    def _calculate_entropy(values: List[int]) -> float:
        """Calculate information entropy of the sequence"""
        if len(values) <= 1:
            return 0.0
            
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        total = len(values)
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def _calculate_autocorrelation(values: List[int], lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(values) <= lag:
            return 0.0
        
        mean = statistics.mean(values)
        numerator = sum((values[i] - mean) * (values[i + lag] - mean) 
                       for i in range(len(values) - lag))
        denominator = sum((x - mean) ** 2 for x in values)
        
        return numerator / denominator if denominator != 0 else 0.0

def enhanced_chaotic_logic(value: int, step: int) -> int:
    """Applies sophisticated chaotic transformations"""
    chaos = SecureRandom.float()
    
    if value % 11 == 0:
        # Major transformation for values divisible by 11
        return value * 3 + SecureRandom.int(41) - 20
    elif value % 7 == 0:
        # Moderate transformation
        return value * 2 + SecureRandom.int(21) - 10
    elif value % 5 == 0:
        # Minor transformation
        return value // 2 + SecureRandom.int(11) - 5
    elif step % 13 == 0:
        # Periodic major disruption
        return value + SecureRandom.int(101) - 50
    elif chaos < 0.1:
        # Random major event (10% chance)
        return value + SecureRandom.int(201) - 100
    else:
        # Normal chaotic adjustment
        return value + SecureRandom.int(21) - 10

class CachedChaoticGenerator:
    """Generator with caching and parallel processing capabilities"""
    
    def __init__(self, config: ChaoticConfig):
        self.config = config
        self.generator = ChaoticSequenceGenerator(config)
    
    @lru_cache(maxsize=100)
    def generate_cached_sequence(self, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate sequence with caching for same parameters"""
        if seed is not None:
            random.seed(seed)
            # Note: secrets can't be seeded, so this is mainly for testing
        
        return self.generator.generate_sequence(n)
    
    def generate_sequence_extended(self, n: int) -> List[Dict[str, Any]]:
        """Generate sequence with enhanced chaotic logic"""
        log = self.generator.generate_sequence(n)

        for i, entry in enumerate(log):
            value = entry["value"]
            enhanced_value = enhanced_chaotic_logic(value, i)
            entry["enhanced_value"] = clamp(enhanced_value, self.config.min_value, self.config.max_value * 2)
            entry["enhancement_delta"] = enhanced_value - value

        return log
    
    def generate_multiple_sequences(self, n: int, count: int) -> List[List[Dict[str, Any]]]:
        """Generate multiple sequences in parallel"""
        with ThreadPoolExecutor() as executor:
            sequences = list(executor.map(
                lambda _: self.generate_sequence_extended(n), 
                range(count)
            ))
        return sequences

class SequenceVisualizer:
    """Visualization tools for chaotic sequences"""
    
    @staticmethod
    def plot_sequence(sequence: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Plot the chaotic sequence with analysis"""
        if not sequence:
            logger.warning("Cannot plot empty sequence")
            return
            
        steps = [entry["step"] for entry in sequence]
        values = [entry["value"] for entry in sequence]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Sequence values
        ax1.plot(steps, values, 'b-', alpha=0.7, linewidth=1, label='Values')
        if "enhanced_value" in sequence[0]:
            enhanced_values = [entry.get("enhanced_value", entry["value"]) for entry in sequence]
            ax1.plot(steps, enhanced_values, 'r-', alpha=0.5, linewidth=1, label='Enhanced Values')
        ax1.set_title('Chaotic Sequence Values')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Value distribution histogram
        ax2.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Value Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Step type distribution
        if HAS_NUMPY:
            types = [entry["type"] for entry in sequence]
            type_counts = pd.Series(types).value_counts()
            ax3.bar(type_counts.index, type_counts.values, alpha=0.7, edgecolor='black')
            ax3.set_title('Step Type Distribution')
            ax3.set_xlabel('Step Type')
            ax3.set_ylabel('Count')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'Pandas required for type analysis', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Step Type Distribution (Pandas required)')
        
        # Plot 4: Cumulative sum
        cumulative_sum = np.cumsum(values) if HAS_NUMPY else []
        if len(cumulative_sum) > 0:
            ax4.plot(steps, cumulative_sum, 'g-', alpha=0.7)
            ax4.set_title('Cumulative Sum')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Cumulative Value')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def export_to_dataframe(sequence: List[Dict[str, Any]]) -> Any:
        """Convert sequence to pandas DataFrame if available"""
        if HAS_NUMPY:
            return pd.DataFrame(sequence)
        else:
            logger.warning("Pandas not available, returning list as is")
            return sequence

class ConfigManager:
    """Configuration management with YAML support"""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> ChaoticConfig:
        """Load configuration from YAML file"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML configuration support")
        
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return ChaoticConfig(**config_data)
    
    @staticmethod
    def save_to_yaml(config: ChaoticConfig, file_path: str):
        """Save configuration to YAML file"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML configuration support")
        
        with open(file_path, 'w') as file:
            yaml.dump(config.__dict__, file, default_flow_style=False)

class JSONEncoderWithNumpy(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if HAS_NUMPY and isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif HAS_NUMPY and isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif HAS_NUMPY and isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_to_json(data: Any, filename: str) -> None:
    """Saves data to a JSON file with proper error handling"""
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2, cls=JSONEncoderWithNumpy)
        logger.info(f"Data successfully saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to create file {filename}: {e}")
        raise
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to encode JSON for {filename}: {e}")
        raise

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description='Generate chaotic sequences with advanced analysis')
    parser.add_argument('--length', type=int, default=50, help='Sequence length (default: 50)')
    parser.add_argument('--config', type=str, help='Config YAML file path')
    parser.add_argument('--output', type=str, default='chaotic_analysis', help='Output file prefix (default: chaotic_analysis)')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    parser.add_argument('--volatility', type=float, help='Volatility parameter (0.0-1.0)')
    parser.add_argument('--trend-strength', type=float, help='Trend strength parameter (0.0-1.0)')
    parser.add_argument('--mean-reversion', type=float, help='Mean reversion parameter (0.0-1.0)')
    parser.add_argument('--min-value', type=int, help='Minimum value')
    parser.add_argument('--max-value', type=int, help='Maximum value')
    
    args = parser.parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            config = ConfigManager.load_from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = ChaoticConfig.default_config()
            logger.info("Using default configuration")
        
        # Override config with command line arguments if provided
        if args.volatility is not None:
            config.volatility = args.volatility
        if args.trend_strength is not None:
            config.trend_strength = args.trend_strength
        if args.mean_reversion is not None:
            config.mean_reversion = args.mean_reversion
        if args.min_value is not None:
            config.min_value = args.min_value
        if args.max_value is not None:
            config.max_value = args.max_value
        
        # Validate final configuration
        config.__post_init__()
        
        # Generate sequence
        logger.info(f"Generating chaotic sequence of length {args.length}")
        generator = CachedChaoticGenerator(config)
        sequence = generator.generate_sequence_extended(args.length)
        
        # Calculate statistics
        values = [entry["value"] for entry in sequence]
        stats = AdvancedStatistics.calculate_advanced_stats(values)
        
        # Print summary
        print("\n" + "="*50)
        print("CHAOTIC SEQUENCE ANALYSIS")
        print("="*50)
        print(f"Generated {len(sequence)} transactions")
        print(f"Value Range: {stats['min']} - {stats['max']}")
        print(f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
        print(f"Std Dev: {stats['stdev']:.2f}, Variance: {stats['variance']:.2f}")
        print(f"Volatility: {stats['volatility']:.2f}, Trend Strength: {stats['trend_strength']:.2f}")
        print(f"IQR: {stats['iqr']:.2f} (Q1: {stats['q1']:.2f}, Q3: {stats['q3']:.2f})")
        print(f"Skewness: {stats['skewness']:.2f}, Kurtosis: {stats['kurtosis']:.2f}")
        print(f"Entropy: {stats['entropy']:.2f}, Autocorrelation: {stats['autocorrelation']:.2f}")
        
        # Save results
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": config.__dict__,
                "sequence_length": len(sequence),
                "analysis_version": "2.0"
            },
            "statistics": stats,
            "sequence": sequence
        }
        
        json_filename = f"{args.output}.json"
        save_to_json(output_data, json_filename)
        
        if args.plot:
            plot_filename = f"{args.output}.png"
            SequenceVisualizer.plot_sequence(sequence, plot_filename)
        
        # Print first few entries as sample
        print(f"\nFirst 5 transactions:")
        sample = sequence[:5]
        for entry in sample:
            print(f"  Step {entry['step']}: {entry['value']} ({entry['type']})")
        
        print(f"\nDetailed analysis saved to {json_filename}")
        if args.plot:
            print(f"Plot saved to {plot_filename}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
