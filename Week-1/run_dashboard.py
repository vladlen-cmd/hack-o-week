import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from predictor import PeakPredictor
from dashboard import ElectricityDashboard


def main():
    """Main function to run the dashboard."""
    
    print("="*60)
    print("⚡ PEAK HOUR ELECTRICITY ANALYSIS")
    print("="*60)
    
    # Initialize data processor
    print("\n📂 Step 1: Loading data...")
    # Load the 7-day dorm energy consumption dataset
    data_path = "data/dorm_energy_7days.csv"
    processor = DataProcessor(data_path)
    processor.load_data()
    
    # Clean and process data
    print("\n🧹 Step 2: Cleaning data...")
    processor.clean_data()
    
    # Apply moving average smoothing
    print("\n📊 Step 3: Applying moving average smoothing...")
    processor.apply_moving_average(window=24)
    
    # Extract features
    print("\n🔧 Step 4: Extracting features...")
    df_features = processor.extract_features()
    
    # Split data
    print("\n✂️  Step 5: Splitting data into train/test sets...")
    train_df, test_df = processor.get_train_test_split(df_features, test_size=0.2)
    
    # Train predictor
    print("\n🤖 Step 6: Training linear regression model...")
    predictor = PeakPredictor()
    predictor.train(train_df)
    
    # Evaluate model
    print("\n📈 Step 7: Evaluating model performance...")
    metrics = predictor.evaluate(test_df)
    
    # Show feature importance
    print("\n🎯 Feature Importance:")
    importance = predictor.get_feature_importance()
    print(importance.head(5).to_string(index=False))
    
    # Initialize and run dashboard
    print("\n🌐 Step 8: Launching dashboard...")
    dashboard = ElectricityDashboard(processor, predictor)
    dashboard.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
