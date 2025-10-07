"""
Pack command for creating run bundles.

This module provides functionality to package backtest results into compressed
archives for easy sharing and storage.
"""

import click
import zipfile
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def create_run_bundle(run_dir: str, output_path: Optional[str] = None) -> str:
    """Create a compressed bundle of a backtest run.
    
    Args:
        run_dir: Path to the run directory
        output_path: Optional output path for the bundle
        
    Returns:
        Path to the created bundle file
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    if not run_path.is_dir():
        raise ValueError(f"Path is not a directory: {run_dir}")
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = run_path.name
        output_path = f"runs/{timestamp}.zip"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the run directory
        for file_path in run_path.rglob('*'):
            if file_path.is_file():
                # Add file with relative path
                arcname = file_path.relative_to(run_path.parent)
                zipf.write(file_path, arcname)
    
    return str(output_path)


def list_run_directories(runs_dir: str = "runs") -> List[str]:
    """List available run directories.
    
    Args:
        runs_dir: Path to the runs directory
        
    Returns:
        List of run directory names, sorted by timestamp (newest first)
    """
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        return []
    
    # Get all directories that look like timestamps
    run_dirs = []
    for item in runs_path.iterdir():
        if item.is_dir() and not item.name.endswith('_sweep'):
            try:
                # Try to parse as timestamp
                datetime.fromisoformat(item.name.replace('T', ' ').replace('-', ':'))
                run_dirs.append(item.name)
            except ValueError:
                continue
    
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    return run_dirs


@click.command()
@click.option('--run', '-r', help='Run directory to pack (timestamp or path)')
@click.option('--output', '-o', help='Output path for the bundle')
@click.option('--list', '-l', 'list_runs', is_flag=True, help='List available runs')
@click.option('--latest', is_flag=True, help='Pack the latest run')
def pack(run: Optional[str], output: Optional[str], list_runs: bool, latest: bool):
    """Pack backtest results into a compressed bundle.
    
    Examples:
        flashback pack --list                    # List available runs
        flashback pack --latest                  # Pack the latest run
        flashback pack --run 2025-10-06T00-00-00 # Pack specific run
        flashback pack --run runs/2025-10-06T00-00-00 --output my_results.zip
    """
    if list_runs:
        runs = list_run_directories()
        if not runs:
            click.echo("No runs found in runs/ directory")
            return
        
        click.echo("Available runs:")
        for run_name in runs:
            run_path = Path("runs") / run_name
            if run_path.exists():
                # Get run info
                config_file = run_path / "config.yaml"
                performance_file = run_path / "performance.json"
                
                info = []
                if config_file.exists():
                    info.append("config")
                if performance_file.exists():
                    info.append("results")
                
                info_str = f" ({', '.join(info)})" if info else ""
                click.echo(f"  {run_name}{info_str}")
        
        return
    
    # Determine which run to pack
    if latest:
        runs = list_run_directories()
        if not runs:
            click.echo("No runs found in runs/ directory")
            return
        run = runs[0]  # Latest run
        click.echo(f"Packing latest run: {run}")
    
    if not run:
        click.echo("Error: Must specify --run, --latest, or --list")
        return
    
    # Resolve run path
    if Path(run).is_absolute() or run.startswith('runs/'):
        run_path = run
    else:
        run_path = f"runs/{run}"
    
    try:
        # Create bundle
        bundle_path = create_run_bundle(run_path, output)
        
        # Get bundle size
        bundle_size = Path(bundle_path).stat().st_size
        bundle_size_mb = bundle_size / (1024 * 1024)
        
        click.echo(f" Created bundle: {bundle_path}")
        click.echo(f" Bundle size: {bundle_size_mb:.2f} MB")
        
        # Show contents
        with zipfile.ZipFile(bundle_path, 'r') as zipf:
            files = zipf.namelist()
            click.echo(f" Contains {len(files)} files:")
            for file in sorted(files)[:10]:  # Show first 10 files
                click.echo(f"   {file}")
            if len(files) > 10:
                click.echo(f"   ... and {len(files) - 10} more files")
        
    except Exception as e:
        click.echo(f" Error creating bundle: {e}")
        raise click.Abort()


def extract_run_bundle(bundle_path: str, extract_dir: Optional[str] = None) -> str:
    """Extract a run bundle.
    
    Args:
        bundle_path: Path to the bundle file
        extract_dir: Directory to extract to (default: runs/)
        
    Returns:
        Path to the extracted directory
    """
    bundle_path = Path(bundle_path)
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    
    if extract_dir is None:
        extract_dir = "runs"
    
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    # Extract bundle
    with zipfile.ZipFile(bundle_path, 'r') as zipf:
        zipf.extractall(extract_path)
    
    # Find the extracted run directory
    for item in extract_path.iterdir():
        if item.is_dir() and not item.name.endswith('_sweep'):
            try:
                # Try to parse as timestamp
                datetime.fromisoformat(item.name.replace('T', ' ').replace('-', ':'))
                return str(item)
            except ValueError:
                continue
    
    raise ValueError("Could not find run directory in extracted bundle")


@click.command()
@click.argument('bundle_path')
@click.option('--extract-dir', '-d', help='Directory to extract to')
def unpack(bundle_path: str, extract_dir: Optional[str]):
    """Extract a backtest run bundle.
    
    Examples:
        flashback unpack results.zip
        flashback unpack results.zip --extract-dir my_runs/
    """
    try:
        extracted_path = extract_run_bundle(bundle_path, extract_dir)
        click.echo(f" Extracted bundle to: {extracted_path}")
        
        # Show contents
        run_path = Path(extracted_path)
        files = list(run_path.iterdir())
        click.echo(f" Extracted {len(files)} items:")
        for file in sorted(files):
            if file.is_file():
                size = file.stat().st_size
                size_str = f" ({size:,} bytes)" if size < 1024 else f" ({size/1024:.1f} KB)"
                click.echo(f"   {file.name}{size_str}")
            else:
                click.echo(f"   {file.name}/")
        
    except Exception as e:
        click.echo(f" Error extracting bundle: {e}")
        raise click.Abort()


if __name__ == "__main__":
    pack()