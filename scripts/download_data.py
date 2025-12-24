"""
Data Download Script
====================

Downloads and prepares real electrical load and weather data.

Data Sources:
1. UCI Electricity Load Diagrams (2011-2014)
2. Open-Meteo Historical Weather API
"""

import pandas as pd
import requests
from pathlib import Path
import zipfile
import io

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("XAI Load Forecasting - Data Download Script")
print("=" * 80)


def download_uci_load_data():
    """Download UCI Electricity Load Diagrams dataset."""
    print("\n[1/3] Downloading UCI Electricity Load Dataset...")

    # Check if already downloaded
    output_file = DATA_RAW / "uci_load_raw.csv"
    if output_file.exists():
        print(f"  ✓ Dataset already exists at: {output_file}")
        print("  Loading existing data...")
        df = pd.read_csv(output_file, index_col=0)
        print(f"  Dataset shape: {df.shape}")
        return df

    try:
        # Try using ucimlrepo package
        from ucimlrepo import fetch_ucirepo

        print("  Fetching dataset using ucimlrepo...")
        dataset = fetch_ucirepo(id=321)

        # Get the data
        X = dataset.data.features

        print(f"  Dataset shape: {X.shape}")
        print(f"  Columns: {list(X.columns[:5])}... (showing first 5)")

        # Save raw data
        output_file = DATA_RAW / "uci_load_raw.csv"
        X.to_csv(output_file)
        print(f"  ✓ Saved to: {output_file}")

        return X

    except ImportError:
        print("  ucimlrepo not installed. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "ucimlrepo"])
        return download_uci_load_data()

    except Exception as e:
        print(f"  ✗ Error downloading UCI dataset: {e}")
        print("  Attempting alternative download method...")

        # Alternative: Direct download
        url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
        print(f"  Downloading from: {url}")

        response = requests.get(url, timeout=300)
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(DATA_RAW)
            print(f"  ✓ Extracted to: {DATA_RAW}")

            # Find and load the CSV file
            csv_files = list(DATA_RAW.glob("*.txt"))
            if csv_files:
                df = pd.read_csv(csv_files[0], sep=";", decimal=",")
                output_file = DATA_RAW / "uci_load_raw.csv"
                df.to_csv(output_file, index=False)
                print(f"  ✓ Saved to: {output_file}")
                return df

        raise Exception("Failed to download UCI dataset")


def process_uci_load_data(df):
    """Process UCI load data to hourly resolution."""
    print("\n[2/3] Processing load data to hourly resolution...")

    # Check if already processed
    output_file = DATA_RAW / "load_data.csv"
    if output_file.exists():
        print(f"  ✓ Processed data already exists at: {output_file}")
        print("  Loading existing data...")
        load_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        print(f"  Processed shape: {load_df.shape}")
        print(f"  Date range: {load_df.index.min()} to {load_df.index.max()}")
        return load_df

    # The UCI dataset has datetime in first column
    if df.index.name is None and len(df.columns) > 0:
        # First column is datetime
        df = df.copy()
        date_col = df.columns[0]

        # Parse datetime
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
        df = df.drop(columns=[date_col])

    print(f"  Original shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Select one client (column) as our feeder load
    # Use a client with good data coverage
    load_column = df.columns[0]  # First client
    load_series = df[load_column].copy()

    # Convert from 15-min to hourly (sum to get kWh)
    # UCI data is in kW for 15-min intervals, divide by 4 to get kWh
    load_series = load_series / 4

    # Resample to hourly (use 'h' instead of deprecated 'H')
    load_hourly = load_series.resample("h").sum()

    # Remove zeros and outliers
    load_hourly = load_hourly[load_hourly > 0]

    # Create DataFrame
    load_df = pd.DataFrame({"load": load_hourly})

    print(f"  Processed shape: {load_df.shape}")
    print("  Load statistics:")
    print(f"    Mean: {load_df['load'].mean():.2f} kW")
    print(f"    Std: {load_df['load'].std():.2f} kW")
    print(f"    Min: {load_df['load'].min():.2f} kW")
    print(f"    Max: {load_df['load'].max():.2f} kW")

    # Save processed data
    output_file = DATA_RAW / "load_data.csv"
    load_df.to_csv(output_file)
    print(f"  ✓ Saved to: {output_file}")

    return load_df


def download_weather_data(start_date, end_date, latitude=41.15, longitude=-8.61):
    """
    Download historical weather data using Open-Meteo API.

    Default location: Porto, Portugal (matches UCI dataset location)
    """
    print("\n[3/3] Downloading weather data from Open-Meteo...")

    # Check if already downloaded
    output_file = DATA_EXTERNAL / "weather.csv"
    if output_file.exists():
        print(f"  ✓ Weather data already exists at: {output_file}")
        print("  Loading existing data...")
        weather_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        print(f"  Weather data shape: {weather_df.shape}")
        print(f"  Date range: {weather_df.index.min()} to {weather_df.index.max()}")
        return weather_df

    print(f"  Location: Lat {latitude}, Lon {longitude}")
    print(f"  Date range: {start_date} to {end_date}")

    # Open-Meteo API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover",
        ],
        "timezone": "UTC",
    }

    print("  Fetching data from Open-Meteo API...")
    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")

    data = response.json()

    # Parse response
    hourly_data = data["hourly"]

    weather_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data["temperature_2m"],
            "humidity": hourly_data["relative_humidity_2m"],
            "precipitation": hourly_data["precipitation"],
            "wind_speed": hourly_data["wind_speed_10m"],
            "wind_direction": hourly_data["wind_direction_10m"],
            "pressure": hourly_data["surface_pressure"],
            "cloud_cover": hourly_data["cloud_cover"],
        }
    )

    weather_df = weather_df.set_index("timestamp")

    print(f"  Weather data shape: {weather_df.shape}")
    print("  Weather statistics:")
    print(
        f"    Temperature: {weather_df['temperature'].mean():.1f}°C (±{weather_df['temperature'].std():.1f})"
    )
    print(
        f"    Humidity: {weather_df['humidity'].mean():.1f}% (±{weather_df['humidity'].std():.1f})"
    )
    print(f"    Wind speed: {weather_df['wind_speed'].mean():.1f} m/s")

    # Save weather data
    output_file = DATA_EXTERNAL / "weather.csv"
    weather_df.to_csv(output_file)
    print(f"  ✓ Saved to: {output_file}")

    return weather_df


def create_sample_dataset():
    """Create a smaller sample dataset for quick testing."""
    print("\n[BONUS] Creating sample dataset (2013 data only)...")

    # Check if already created
    output_file = DATA_RAW / "sample_data_2013.csv"
    if output_file.exists():
        print(f"  ✓ Sample dataset already exists at: {output_file}")
        sample_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        print(f"  Sample dataset shape: {sample_df.shape}")
        return sample_df

    # Load full datasets
    load_df = pd.read_csv(DATA_RAW / "load_data.csv", index_col=0, parse_dates=True)
    weather_df = pd.read_csv(
        DATA_EXTERNAL / "weather.csv", index_col=0, parse_dates=True
    )

    # Filter to 2013 only (use loc with date range)
    load_sample = load_df.loc["2013-01-01":"2013-12-31"]
    weather_sample = weather_df.loc["2013-01-01":"2013-12-31"]

    # Merge
    sample_df = load_sample.merge(
        weather_sample, left_index=True, right_index=True, how="inner"
    )

    print(f"  Sample dataset shape: {sample_df.shape}")
    print(f"  Date range: {sample_df.index.min()} to {sample_df.index.max()}")

    # Save sample
    output_file = DATA_RAW / "sample_data_2013.csv"
    sample_df.to_csv(output_file)
    print(f"  ✓ Saved to: {output_file}")

    return sample_df


def main():
    """Main execution function."""
    try:
        # Step 1: Download UCI load data
        uci_data = download_uci_load_data()

        # Step 2: Process to hourly
        load_df = process_uci_load_data(uci_data)

        # Step 3: Download weather data for same period
        start_date = load_df.index.min()
        end_date = load_df.index.max()

        weather_df = download_weather_data(start_date, end_date)

        # Step 4: Create sample dataset
        sample_df = create_sample_dataset()

        print("\n" + "=" * 80)
        print("✓ DATA DOWNLOAD COMPLETE")
        print("=" * 80)
        print("\nAvailable datasets:")
        print("  1. Full load data: data/raw/load_data.csv")
        print("  2. Weather data: data/external/weather.csv")
        print("  3. Sample (2013): data/raw/sample_data_2013.csv")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook notebooks/01_data_profiling.ipynb")
        print("  2. Or use the sample dataset for quick testing")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
