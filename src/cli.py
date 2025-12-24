"""Command-line interface for XAI Load Forecasting."""

import click
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """XAI Load Forecasting - Day-ahead load prediction with explainability."""
    pass


@cli.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
def profile(config):
    """Run data profiling on raw data."""
    from scripts.run_data_profiling import main

    click.echo("Running data profiling...")
    exit_code = main()
    sys.exit(exit_code)


@cli.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
def features(config):
    """Generate features from processed data."""
    from scripts.run_feature_engineering import main

    click.echo("Running feature engineering...")
    exit_code = main()
    sys.exit(exit_code)


@cli.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
def train(config):
    """Train the forecasting model."""
    from scripts.run_model_training import main

    click.echo("Training model...")
    exit_code = main()
    sys.exit(exit_code)


@cli.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
def explain(config):
    """Run SHAP analysis on trained model."""
    from scripts.run_shap_analysis import main

    click.echo("Running SHAP analysis...")
    exit_code = main()
    sys.exit(exit_code)


@cli.command()
@click.option("--model", "-m", required=True, help="Path to trained model")
@click.option("--data", "-d", required=True, help="Path to input data (CSV/Parquet)")
@click.option("--output", "-o", default=None, help="Output file for predictions")
@click.option("--horizon", "-h", default=24, help="Forecast horizon in hours")
def predict(model, data, output, horizon):
    """Generate predictions from a trained model."""
    import pandas as pd
    from src.models.gbm import GradientBoostingModel

    click.echo(f"Loading model from {model}...")
    gbm = GradientBoostingModel.load(model)

    click.echo(f"Loading data from {data}...")
    if data.endswith(".parquet"):
        df = pd.read_parquet(data)
    else:
        df = pd.read_csv(data, index_col=0, parse_dates=True)

    # Get feature columns that match model
    feature_cols = [c for c in df.columns if c in gbm.feature_names]
    if len(feature_cols) < len(gbm.feature_names):
        missing = set(gbm.feature_names) - set(feature_cols)
        click.echo(f"Warning: Missing features: {missing}", err=True)

    X = df[feature_cols]
    predictions = gbm.predict(X)

    result = pd.DataFrame({"timestamp": df.index, "prediction": predictions})

    if output:
        result.to_csv(output, index=False)
        click.echo(f"Predictions saved to {output}")
    else:
        click.echo(result.to_string())


@cli.command()
def test():
    """Run the test suite."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"], cwd=PROJECT_ROOT
    )
    sys.exit(result.returncode)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def serve(host, port):
    """Start the prediction API server."""
    try:
        import uvicorn

        click.echo(f"Starting API server on {host}:{port}...")
        uvicorn.run("src.api:app", host=host, port=port, reload=True)
    except ImportError:
        click.echo("Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
