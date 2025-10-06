"""Packaging utility for backtest runs."""

import argparse
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RunPacker:
    """Utility for packaging backtest run directories."""
    
    def __init__(self):
        """Initialize the run packer."""
        self.logger = logging.getLogger(__name__)
    
    def pack_run(self, run_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Pack a run directory into a zip file.
        
        Args:
            run_path: Path to the run directory
            output_path: Optional output path for the zip file
            
        Returns:
            Path to the created zip file
        """
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        if not run_path.is_dir():
            raise ValueError(f"Path is not a directory: {run_path}")
        
        # Determine output path
        if output_path is None:
            output_path = run_path.parent / f"{run_path.name}.zip"
        
        self.logger.info(f"Packing run directory: {run_path}")
        self.logger.info(f"Output zip file: {output_path}")
        
        # Create zip file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            self._add_directory_to_zip(zipf, run_path, run_path.name)
        
        # Verify the zip file was created
        if not output_path.exists():
            raise RuntimeError(f"Failed to create zip file: {output_path}")
        
        # Log file sizes
        run_size = self._get_directory_size(run_path)
        zip_size = output_path.stat().st_size
        compression_ratio = (1 - zip_size / run_size) * 100 if run_size > 0 else 0
        
        self.logger.info(f"Packing completed successfully")
        self.logger.info(f"Original size: {self._format_size(run_size)}")
        self.logger.info(f"Compressed size: {self._format_size(zip_size)}")
        self.logger.info(f"Compression ratio: {compression_ratio:.1f}%")
        
        return output_path
    
    def _add_directory_to_zip(self, zipf: zipfile.ZipFile, directory: Path, arcname: str):
        """Recursively add directory contents to zip file."""
        for item in directory.rglob('*'):
            if item.is_file():
                # Calculate relative path for archive
                arc_path = item.relative_to(directory.parent)
                zipf.write(item, arc_path)
                self.logger.debug(f"Added: {arc_path}")
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory contents."""
        total_size = 0
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def list_run_contents(self, run_path: Path) -> dict:
        """
        List contents of a run directory.
        
        Args:
            run_path: Path to the run directory
            
        Returns:
            Dictionary with file information
        """
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        contents = {
            'files': [],
            'total_size': 0,
            'file_count': 0
        }
        
        for item in run_path.rglob('*'):
            if item.is_file():
                file_info = {
                    'path': str(item.relative_to(run_path)),
                    'size': item.stat().st_size,
                    'size_formatted': self._format_size(item.stat().st_size)
                }
                contents['files'].append(file_info)
                contents['total_size'] += file_info['size']
                contents['file_count'] += 1
        
        contents['total_size_formatted'] = self._format_size(contents['total_size'])
        return contents
    
    def validate_run_directory(self, run_path: Path) -> dict:
        """
        Validate that a run directory contains all required files.
        
        Args:
            run_path: Path to the run directory
            
        Returns:
            Dictionary with validation results
        """
        required_files = [
            'config.yaml',
            'performance.json',
            'performance.csv',
            'blotter.parquet',
            'positions.parquet'
        ]
        
        optional_files = [
            'equity_curve.png',
            'drawdown.png',
            'trade_pnl_hist.png',
            'latency_sweep.csv'
        ]
        
        validation = {
            'valid': True,
            'missing_required': [],
            'missing_optional': [],
            'present_files': [],
            'errors': []
        }
        
        if not run_path.exists():
            validation['valid'] = False
            validation['errors'].append(f"Directory does not exist: {run_path}")
            return validation
        
        if not run_path.is_dir():
            validation['valid'] = False
            validation['errors'].append(f"Path is not a directory: {run_path}")
            return validation
        
        # Check required files
        for file_name in required_files:
            file_path = run_path / file_name
            if file_path.exists():
                validation['present_files'].append(file_name)
            else:
                validation['missing_required'].append(file_name)
                validation['valid'] = False
        
        # Check optional files
        for file_name in optional_files:
            file_path = run_path / file_name
            if file_path.exists():
                validation['present_files'].append(file_name)
            else:
                validation['missing_optional'].append(file_name)
        
        return validation


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Pack backtest run directories")
    parser.add_argument("--run", required=True, type=Path, 
                       help="Path to run directory (e.g., runs/2025-10-06T00-00-00)")
    parser.add_argument("--output", "-o", type=Path, 
                       help="Output zip file path (default: run_directory.zip)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List contents of run directory")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Validate run directory structure")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    packer = RunPacker()
    
    try:
        if args.list:
            # List contents
            contents = packer.list_run_contents(args.run)
            print(f"\nContents of {args.run}:")
            print(f"Total files: {contents['file_count']}")
            print(f"Total size: {contents['total_size_formatted']}")
            print("\nFiles:")
            for file_info in contents['files']:
                print(f"  {file_info['path']:<30} {file_info['size_formatted']:>10}")
        
        elif args.validate:
            # Validate structure
            validation = packer.validate_run_directory(args.run)
            print(f"\nValidation of {args.run}:")
            print(f"Valid: {validation['valid']}")
            
            if validation['present_files']:
                print(f"\nPresent files ({len(validation['present_files'])}):")
                for file_name in validation['present_files']:
                    print(f"  ✓ {file_name}")
            
            if validation['missing_required']:
                print(f"\nMissing required files ({len(validation['missing_required'])}):")
                for file_name in validation['missing_required']:
                    print(f"  ✗ {file_name}")
            
            if validation['missing_optional']:
                print(f"\nMissing optional files ({len(validation['missing_optional'])}):")
                for file_name in validation['missing_optional']:
                    print(f"  ? {file_name}")
            
            if validation['errors']:
                print(f"\nErrors:")
                for error in validation['errors']:
                    print(f"  ✗ {error}")
        
        else:
            # Pack the run
            output_path = packer.pack_run(args.run, args.output)
            print(f"\nSuccessfully packed run to: {output_path}")
            
            # Show validation
            validation = packer.validate_run_directory(args.run)
            if validation['valid']:
                print("✓ Run directory is valid")
            else:
                print("⚠ Run directory has missing files:")
                for file_name in validation['missing_required']:
                    print(f"  ✗ {file_name}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
