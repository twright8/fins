"""
Utility for profiling application performance during runtime.
Can be used to profile different aspects of the application,
including import times, function execution times, and more.
"""
import os
import sys
import time
import inspect
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the import profiler
from src.utils.import_profiler import ImportProfiler

class AppProfiler:
    """
    A utility class to profile various aspects of the application.
    """
    
    def __init__(self, 
                 enabled: bool = True, 
                 log_to_console: bool = True, 
                 log_to_file: Optional[str] = None,
                 profile_folder: str = 'logs/profiles'):
        """
        Initialize the application profiler.
        
        Args:
            enabled (bool): Whether profiling is enabled
            log_to_console (bool): Whether to print results to console
            log_to_file (str, optional): File path to log results, or None to auto-generate
            profile_folder (str): Folder to store profiling logs
        """
        self.enabled = enabled
        self.log_to_console = log_to_console
        
        # Set up logging directory
        self.profile_folder = Path(project_root) / profile_folder
        os.makedirs(self.profile_folder, exist_ok=True)
        
        # Set up log file
        if log_to_file:
            self.log_file = log_to_file
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file = str(self.profile_folder / f"app_profile_{timestamp}.txt")
        
        # Initialize the import profiler
        self.import_profiler = ImportProfiler(
            log_to_console=log_to_console,
            log_to_file=self.log_file
        )
        
        # Initialize function timing stats
        self.function_stats = {}
        
        if self.enabled and self.log_to_console:
            print(f"Application profiling enabled. Results will be saved to {self.log_file}")
    
    def profile_imports(self, modules: Optional[List[str]] = None) -> None:
        """
        Profile import times for modules.
        
        Args:
            modules (list, optional): List of module names to import, or None to use defaults
        """
        if not self.enabled:
            return
            
        if modules is None:
            # Default modules to profile
            modules = [
                # Core Python
                "os", "sys", "time", "pathlib", "multiprocessing", "concurrent.futures",
                
                # ML Libraries
                "numpy", "torch", "transformers", "sentence_transformers",
                
                # Document Processing
                "fitz", "docx", "pandas", "PIL", 
                
                # NLP
                "nltk", "flair.nn", "fastcoref",
                
                # LangChain
                "langchain", "langchain_huggingface",
                
                # Vector DB
                "qdrant_client", "rank_bm25"
            ]
        
        start_time = time.time()
        self.import_profiler.profile_imports(modules)
        total_time = time.time() - start_time
        
        if self.log_to_console:
            self.import_profiler.print_summary()
            print(f"\nImport profiling completed in {total_time:.2f} seconds")
    
    def profile_function(self, func: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
        """
        Decorator to profile function execution time.
        
        Args:
            func (callable, optional): Function to profile
            name (str, optional): Custom name for the function in logs
            
        Returns:
            callable: Decorated function
        """
        def decorator(fn):
            fn_name = name or fn.__name__
            
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)
                
                # Get source info
                try:
                    source_file = inspect.getsourcefile(fn)
                    source_line = inspect.getsourcelines(fn)[1]
                    source_info = f"{source_file}:{source_line}"
                except (IOError, TypeError):
                    source_info = "unknown"
                
                # Record function entry
                start_time = time.time()
                if self.log_to_console:
                    print(f"[PROFILE] Starting {fn_name}...")
                
                # Execute function
                try:
                    result = fn(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    exception = e
                    
                # Record function exit
                elapsed = time.time() - start_time
                
                # Update stats
                if fn_name not in self.function_stats:
                    self.function_stats[fn_name] = {
                        "calls": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0,
                        "source": source_info
                    }
                
                stats = self.function_stats[fn_name]
                stats["calls"] += 1
                stats["total_time"] += elapsed
                stats["min_time"] = min(stats["min_time"], elapsed)
                stats["max_time"] = max(stats["max_time"], elapsed)
                
                # Log result
                if self.log_to_console:
                    print(f"[PROFILE] {fn_name} completed in {elapsed:.4f}s")
                
                # Write to file
                with open(self.log_file, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {fn_name} - {elapsed:.4f}s\n")
                
                # Handle result or exception
                if success:
                    return result
                else:
                    raise exception
                    
            return wrapper
        
        # Handle both @profile_function and @profile_function()
        if func is None:
            return decorator
        return decorator(func)
    
    def print_function_stats(self) -> None:
        """Print a summary of function execution times."""
        if not self.enabled or not self.function_stats:
            return
            
        print("\n=== Function Profiling Summary ===")
        
        # Sort by total time spent
        sorted_stats = sorted(
            self.function_stats.items(),
            key=lambda item: item[1]["total_time"],
            reverse=True
        )
        
        # Print as a table
        print("\n┌────────────────────────┬─────────┬────────────┬────────────┬────────────┬──────────────────────────────────┐")
        print("│ Function                │  Calls  │ Total Time │  Min Time  │  Max Time  │ Source                            │")
        print("├────────────────────────┼─────────┼────────────┼────────────┼────────────┼──────────────────────────────────┤")
        
        for name, data in sorted_stats:
            name_trimmed = name[:22].ljust(22)
            calls = str(data["calls"]).rjust(7)
            total = f"{data['total_time']:.4f}s".rjust(10)
            min_t = f"{data['min_time']:.4f}s".rjust(10)
            max_t = f"{data['max_time']:.4f}s".rjust(10)
            source = data["source"][:30].ljust(34)
            print(f"│ {name_trimmed} │ {calls} │ {total} │ {min_t} │ {max_t} │ {source} │")
            
        print("└────────────────────────┴─────────┴────────────┴────────────┴────────────┴──────────────────────────────────┘")
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write("\n=== Function Profiling Summary ===\n")
            for name, data in sorted_stats:
                f.write(f"{name}: {data['calls']} calls, {data['total_time']:.4f}s total, " 
                       f"{data['min_time']:.4f}s min, {data['max_time']:.4f}s max\n")


# Example usage
if __name__ == "__main__":
    profiler = AppProfiler()
    
    # Profile imports
    profiler.profile_imports(["numpy", "pandas", "matplotlib"])
    
    # Example function profiling
    @profiler.profile_function
    def slow_function(n):
        time.sleep(n)
        return n * 2
    
    slow_function(0.5)
    slow_function(1.0)
    
    # Print function stats
    profiler.print_function_stats()
