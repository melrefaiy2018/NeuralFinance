#!/usr/bin/env python3
"""
Utility script to organize scattered output files from stock prediction analysis.

This script helps clean up files that were saved outside of organized directories
and moves them into a proper calculations directory structure.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import glob

def find_scattered_files():
    """
    Find files that might have been created by stock prediction analysis.
    
    Returns:
        dict: Dictionary of file types and their paths
    """
    current_dir = Path.cwd()
    scattered_files = {
        'predictions': [],
        'charts': [],
        'models': [],
        'data': [],
        'logs': [],
        'other': []
    }
    
    # Common patterns for files created by stock analysis
    patterns = {
        'predictions': ['*predictions*.txt', '*prediction*.csv', '*future*.txt'],
        'charts': ['*.png', '*.jpg', '*.pdf', '*plot*.html'],
        'models': ['*.h5', '*.pkl', '*.model', '*model*'],
        'data': ['*_data.csv', '*stock*.csv', '*sentiment*.csv'],
        'logs': ['*.log', '*error*.txt', '*debug*.txt'],
    }
    
    print("🔍 Searching for scattered files...")
    
    for file_type, file_patterns in patterns.items():
        for pattern in file_patterns:
            matches = list(current_dir.glob(pattern))
            for match in matches:
                if match.is_file() and not match.name.startswith('.'):
                    scattered_files[file_type].append(match)
                    print(f"   Found {file_type}: {match.name}")
    
    # Check for any other analysis-related files
    analysis_keywords = ['AAPL', 'NVDA', 'TSLA', 'stock', 'prediction', 'analysis']
    for file in current_dir.iterdir():
        if file.is_file() and not file.name.startswith('.'):
            if any(keyword.lower() in file.name.lower() for keyword in analysis_keywords):
                if not any(file in files for files in scattered_files.values()):
                    scattered_files['other'].append(file)
                    print(f"   Found other: {file.name}")
    
    total_files = sum(len(files) for files in scattered_files.values())
    print(f"📊 Found {total_files} scattered files")
    
    return scattered_files

def create_organized_directory():
    """
    Create an organized directory structure for cleaning up files.
    
    Returns:
        Path: Path to the created organization directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    org_dir = Path(f"organized_files_{timestamp}")
    org_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "predictions",
        "charts", 
        "models",
        "data",
        "logs",
        "other",
        "original_locations"  # To keep track of where files came from
    ]
    
    for subdir in subdirs:
        (org_dir / subdir).mkdir(exist_ok=True)
    
    print(f"📁 Created organization directory: {org_dir.absolute()}")
    return org_dir

def move_files_safely(scattered_files, org_dir):
    """
    Move scattered files to organized directory structure.
    
    Args:
        scattered_files (dict): Dictionary of scattered files by type
        org_dir (Path): Organization directory path
    """
    moved_files = []
    file_mapping = []
    
    print("\n📦 Moving files to organized structure...")
    
    for file_type, files in scattered_files.items():
        if not files:
            continue
            
        dest_dir = org_dir / file_type
        print(f"\n📂 Processing {file_type} files:")
        
        for file_path in files:
            try:
                # Create unique filename if conflict exists
                dest_file = dest_dir / file_path.name
                counter = 1
                while dest_file.exists():
                    name_parts = file_path.name.rsplit('.', 1)
                    if len(name_parts) == 2:
                        dest_file = dest_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        dest_file = dest_dir / f"{file_path.name}_{counter}"
                    counter += 1
                
                # Move the file
                shutil.move(str(file_path), str(dest_file))
                moved_files.append(dest_file)
                file_mapping.append((str(file_path), str(dest_file)))
                print(f"   ✅ {file_path.name} → {file_type}/{dest_file.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to move {file_path.name}: {str(e)}")
    
    # Save file mapping for reference
    mapping_file = org_dir / "original_locations" / "file_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("File Organization Mapping\n")
        f.write("=" * 30 + "\n")
        f.write(f"Organized on: {datetime.now()}\n\n")
        f.write("Original Location → New Location\n")
        f.write("-" * 50 + "\n")
        for original, new in file_mapping:
            f.write(f"{original} → {new}\n")
    
    print(f"\n📋 File mapping saved to: {mapping_file}")
    print(f"📊 Successfully moved {len(moved_files)} files")
    
    return moved_files

def create_cleanup_summary(org_dir, moved_files):
    """
    Create a summary of the cleanup operation.
    
    Args:
        org_dir (Path): Organization directory path
        moved_files (list): List of moved file paths
    """
    summary_file = org_dir / "cleanup_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Stock Prediction LSTM - File Cleanup Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Cleanup completed: {datetime.now()}\n")
        f.write(f"Organization directory: {org_dir.absolute()}\n")
        f.write(f"Total files organized: {len(moved_files)}\n\n")
        
        # Count files by type
        f.write("Files by category:\n")
        f.write("-" * 20 + "\n")
        for subdir in org_dir.iterdir():
            if subdir.is_dir() and subdir.name != "original_locations":
                file_count = len(list(subdir.glob("*")))
                f.write(f"{subdir.name}: {file_count} files\n")
        
        f.write(f"\nFiles organized into:\n")
        f.write(f"  📁 {org_dir}/predictions/     # Prediction results\n")
        f.write(f"  📁 {org_dir}/charts/          # Visualization files\n") 
        f.write(f"  📁 {org_dir}/models/          # Saved models\n")
        f.write(f"  📁 {org_dir}/data/            # Data files\n")
        f.write(f"  📁 {org_dir}/logs/            # Log files\n")
        f.write(f"  📁 {org_dir}/other/           # Other analysis files\n")
        f.write(f"  📁 {org_dir}/original_locations/ # File mapping reference\n")
    
    print(f"📄 Cleanup summary saved to: {summary_file}")

def main():
    """
    Main function to organize scattered files.
    """
    print("🧹 Stock Prediction LSTM - File Organization Tool")
    print("=" * 60)
    print("This tool helps organize scattered output files into a clean directory structure.")
    print("")
    
    # Find scattered files
    scattered_files = find_scattered_files()
    
    # Check if any files were found
    total_files = sum(len(files) for files in scattered_files.values())
    if total_files == 0:
        print("✅ No scattered files found! Your directory is already clean.")
        return
    
    # Ask user for confirmation
    print(f"\n📋 Found {total_files} files that can be organized.")
    response = input("Do you want to organize these files? (y/n): ").lower().strip()
    
    if response != 'y':
        print("❌ Organization cancelled by user.")
        return
    
    # Create organized directory
    org_dir = create_organized_directory()
    
    # Move files
    moved_files = move_files_safely(scattered_files, org_dir)
    
    # Create summary
    create_cleanup_summary(org_dir, moved_files)
    
    print("\n" + "=" * 60)
    print("🎉 File Organization Complete!")
    print("=" * 60)
    print(f"📁 Files organized in: {org_dir.absolute()}")
    print(f"📊 {len(moved_files)} files moved successfully")
    print("\n💡 Future tip: Use the updated examples/basic_usage.py")
    print("   which automatically creates organized directories!")

if __name__ == "__main__":
    main()
