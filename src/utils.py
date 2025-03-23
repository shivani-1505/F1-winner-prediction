def is_running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_base_path():
    if is_running_on_colab():
        return "/content/f1-prediction/"
    else:
        return "./"  # Current directory relative path

def setup_environment():
    """Install required packages if needed"""
    import sys
    import importlib.util
    
    required_packages = ['pandas', 'numpy', 'matplotlib', 'sklearn', 'requests']
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            if is_running_on_colab():
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Installed {package}")
            else:
                print(f"Warning: {package} is not installed. Please install it manually.")