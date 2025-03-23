import requests
import pandas as pd
import os
import sys
from pathlib import Path

# Add the parent directory to path to import utils when run as script
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_base_path

def get_race_data(year):
    """Get race results for a specific season"""
    url = f"http://ergast.com/api/f1/{year}/results.json?limit=1000"
    response = requests.get(url)
    data = response.json()
    results = []
    for race in data['MRData']['RaceTable']['Races']:
        race_name = race['raceName']
        circuit = race['Circuit']['circuitName']

        for result in race['Results']:
            driver = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
            team = result['Constructor']['name']
            position = int(result['position'])

            results.append({
                'race': race_name,
                'circuit': circuit,
                'driver': driver,
                'team': team,
                'position': position,
                'points': float(result['points'])
            })

    return pd.DataFrame(results)

def get_qualifying_data(year):
    """Get qualifying results"""
    url = f"http://ergast.com/api/f1/{year}/qualifying.json?limit=1000"
    response = requests.get(url)
    data = response.json()

    results = []
    for race in data['MRData']['RaceTable']['Races']:
        race_name = race['raceName']

        for result in race.get('QualifyingResults', []):
            driver = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
            team = result['Constructor']['name']
            position = int(result['position'])

            results.append({
                'race': race_name,
                'driver': driver,
                'team': team,
                'qualifying_position': position
            })

    return pd.DataFrame(results)

def collect_and_save_data(year=2022, visualize=True):
    """Collect data and save to CSV"""
    print(f"Fetching F1 data for {year}...")
    
    race_df = get_race_data(year)
    qual_df = get_qualifying_data(year)

    # Merge qualifying and race data
    f1_data = pd.merge(qual_df, race_df, on=['race', 'driver', 'team'])
    
    # Create data directory if it doesn't exist
    base_path = get_base_path()
    data_path = os.path.join(base_path, "data", "raw")
    os.makedirs(data_path, exist_ok=True)
    
    # Save to CSV
    f1_data.to_csv(os.path.join(data_path, f'f1_data_{year}.csv'), index=False)
    
    print("Data collection complete!")
    print(f"Total races: {f1_data['race'].nunique()}")
    print(f"Total drivers: {f1_data['driver'].nunique()}")
    
    if visualize:
        import matplotlib.pyplot as plt
        
        # Create results directory if it doesn't exist
        results_path = os.path.join(base_path, "results")
        os.makedirs(results_path, exist_ok=True)
        
        # Basic visualizations
        plt.figure(figsize=(10, 6))
        plt.scatter(f1_data['qualifying_position'], f1_data['position'])
        plt.xlabel('Qualifying Position')
        plt.ylabel('Race Finish Position')
        plt.title('Qualifying Position vs Race Result')
        plt.grid(True)
        plt.savefig(os.path.join(results_path, 'qual_vs_race.png'))
        plt.close()

        # Show top drivers by points
        top_drivers = race_df.groupby('driver')['points'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        top_drivers.plot(kind='bar')
        plt.title('Top 10 Drivers by Points')
        plt.ylabel('Points')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'top_drivers.png'))
        plt.close()
    
    return f1_data

if __name__ == "__main__":
    collect_and_save_data(year=2022)