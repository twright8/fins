"""
Utility for profiling import times of Python modules.
Helps identify slow imports in the application.
"""
import time
import importlib
import sys
from typing import Dict, List, Optional, Tuple, Union

class ImportProfiler:
    """A utility class to profile import times for Python modules."""
    
    def __init__(self, log_to_console: bool = True, log_to_file: Optional[str] = None):
        """
        Initialize the import profiler.
        
        Args:
            log_to_console (bool): Whether to print results to console
            log_to_file (str, optional): File path to log results, or None to disable file logging
        """
        self.results = {}
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.total_time = 0.0
    
    def profile_import(self, module_name: str, alias: Optional[str] = None) -> Tuple[bool, float]:
        """
        Profile the import time for a single module.
        
        Args:
            module_name (str): Name of the module to import
            alias (str, optional): Alias to use for the module in results
            
        Returns:
            tuple: (success, elapsed_time)
        """
        name = alias or module_name
        start = time.time()
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                elapsed = time.time() - start
                self.results[name] = {
                    "success": True,
                    "time": elapsed,
                    "status": "already imported"
                }
                message = f"Module {module_name} was already imported ({elapsed:.4f}s)"
            else:
                module = importlib.import_module(module_name)
                elapsed = time.time() - start
                self.results[name] = {
                    "success": True,
                    "time": elapsed,
                    "status": "imported successfully"
                }
                message = f"Imported {module_name} in {elapsed:.4f}s"
                
            if self.log_to_console:
                print(message)
            
            self.total_time += elapsed
            return True, elapsed
            
        except Exception as e:
            elapsed = time.time() - start
            self.results[name] = {
                "success": False,
                "time": elapsed,
                "status": f"error: {e}"
            }
            message = f"Error importing {module_name} after {elapsed:.4f}s: {e}"
            
            if self.log_to_console:
                print(message)
                
            self.total_time += elapsed
            return False, elapsed
    
    def profile_imports(self, modules: List[str]) -> Dict[str, Dict]:
        """
        Profile import times for multiple modules.
        
        Args:
            modules (list): List of module names to import
        
        Returns:
            dict: Results dictionary with timings
        """
        for module_name in modules:
            self.profile_import(module_name)
            
        return self.results
    
    def profile_nested_imports(self, modules_dict: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Profile import times for nested modules.
        
        Args:
            modules_dict (dict): Dictionary of module groups with lists of module names
            
        Returns:
            dict: Results dictionary with timings
        """
        for group_name, module_list in modules_dict.items():
            if self.log_to_console:
                print(f"\n--- Profiling {group_name} imports ---")
                
            group_start = time.time()
            for module_name in module_list:
                self.profile_import(module_name)
            group_time = time.time() - group_start
            
            if self.log_to_console:
                print(f"--- {group_name} total: {group_time:.4f}s ---")
                
        return self.results
        
    def print_summary(self) -> None:
        """Print a summary of import times."""
        print("\n=== Import Profiling Summary ===")
        
        # Sort results by import time (descending)
        sorted_results = sorted(
            self.results.items(), 
            key=lambda item: item[1]["time"], 
            reverse=True
        )
        
        # Print individual module times
        print("\nModule Import Times (sorted by duration):")
        print("┌────────────────────────────────┬────────────┬───────────────────────┐")
        print("│ Module                         │ Time (sec) │ Status                │")
        print("├────────────────────────────────┼────────────┼───────────────────────┤")
        
        for name, result in sorted_results:
            status = result["status"]
            time_str = f"{result['time']:.4f}"
            name_trimmed = name[:28].ljust(28)
            time_trimmed = time_str.ljust(10)
            status_trimmed = status[:21].ljust(21)
            print(f"│ {name_trimmed} │ {time_trimmed} │ {status_trimmed} │")
            
        print("└────────────────────────────────┴────────────┴───────────────────────┘")
        
        # Print total time
        print(f"\nTotal profiling time: {self.total_time:.4f} seconds")
        
        # Write to file if requested
        if self.log_to_file:
            self._write_to_file()
    
    def _write_to_file(self) -> None:
        """Write profiling results to file."""
        try:
            with open(self.log_to_file, 'w') as f:
                f.write("=== Import Profiling Results ===\n\n")
                
                # Sort results by import time (descending)
                sorted_results = sorted(
                    self.results.items(), 
                    key=lambda item: item[1]["time"], 
                    reverse=True
                )
                
                # Write individual module times
                f.write("Module Import Times (sorted by duration):\n")
                for name, result in sorted_results:
                    f.write(f"{name}: {result['time']:.4f}s - {result['status']}\n")
                
                # Write total time
                f.write(f"\nTotal profiling time: {self.total_time:.4f} seconds\n")
                
        except Exception as e:
            print(f"Error writing to log file: {e}")

# Example usage
if __name__ == "__main__":
    profiler = ImportProfiler()
    
    # Profile some common Python modules
    common_modules = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn"
    ]
    
    profiler.profile_imports(common_modules)
    profiler.print_summary()
