import os, sys
from pathlib import Path


def setup_r_environment():
    """
    Setup R environment with automatic path detection.
    
    This function attempts to locate R installation automatically and
    sets up the necessary environment variables for rpy2 to work.
    
    Raises:
        RuntimeError: If R installation cannot be found
    """
    if 'R_HOME' not in os.environ:
        # Try common R installation paths on Windows
        possible_paths = [
            r'C:\Program Files\R\R-4.5.1',
            r'C:\Program Files\R\R-4.5.0',
            r'C:\Program Files\R\R-4.4.3',
            r'C:\Program Files\R\R-4.4.2',
            r'C:\Program Files\R\R-4.4.1',
            r'C:\Program Files\R\R-4.4.0',
            r'C:\Program Files\R\R-4.3.3',
            r'C:\Program Files\R\R-4.3.2', 
            r'C:\Program Files\R\R-4.3.1',
            r'C:\Program Files\R\R-4.3.0',
            r'C:\Program Files\R\R-4.2.3',
            r'C:\Program Files\R\R-4.2.2',
            r'C:\Program Files\R\R-4.2.1',
            r'C:\Program Files\R\R-4.2.0',
            r'C:\Program Files\R\R-4.1.3',
            # Alternative installation paths
            r'C:\Program Files (x86)\R\R-4.4.1',
            r'C:\Program Files (x86)\R\R-4.3.3',
            r'C:\Users\%USERNAME%\Documents\R\R-4.4.1',
            # Unix/Linux paths (in case this runs on other systems)
            '/usr/lib/R',
            '/usr/local/lib/R',
            '/opt/R',
        ]
        
        r_home = None
        for path in possible_paths:
            # Expand environment variables like %USERNAME%
            expanded_path = os.path.expandvars(path)
            if Path(expanded_path).exists():
                r_home = expanded_path
                break
        
        if r_home is None:
            raise RuntimeError(
                "R installation not found. Please either:\n"
                "1. Install R from https://cran.r-project.org/\n" 
                "2. Set R_HOME environment variable manually\n"
                "3. Add R to your system PATH\n\n"
                f"Searched in the following locations:\n" + 
                "\n".join([f"  - {os.path.expandvars(p)}" for p in possible_paths])
            )
        
        os.environ['R_HOME'] = r_home
        
        # Set up PATH for R binaries
        if sys.platform.startswith('win'):
            bin_path = Path(r_home) / 'bin' / 'x64'
        else:
            bin_path = Path(r_home) / 'bin'
            
        if bin_path.exists():
            os.environ['PATH'] += os.pathsep + str(bin_path)
        
        print(f"Using R installation at: {r_home}")
    else:
        print(f"Using existing R_HOME: {os.environ['R_HOME']}")