from src.utils import setup_environment, get_base_path
import os

def main():
    """Main entry point for the F1 prediction pipeline"""
    # Setup environment
    setup_environment()
    
    # Create necessary directories
    base_path = get_base_path()
    for directory in ['data/raw', 'models', 'results']:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    # Collect data
    from src.data.collect import collect_and_save_data
    year = 2022  # You can change this or make it a parameter
    f1_data = collect_and_save_data(year=year)
    
    # Train model
    from src.models.basic_model import train_basic_model
    model_results = train_basic_model(year=year)
    
    print(f"\nPipeline completed successfully! Model accuracy: {model_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()