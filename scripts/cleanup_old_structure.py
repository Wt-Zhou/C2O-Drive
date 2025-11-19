#!/usr/bin/env python3
"""Script to clean up old directory structure after migration to new architecture."""

import os
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime


def get_folders_to_clean():
    """Identify folders that can be cleaned up after migration."""
    folders_to_clean = {
        "old_tests": [
            "tests/",  # Old test directory (migrated to src/c2o_drive/tests/)
            "carla_c2osr/tests/",  # Old CARLA tests (migrated)
        ],
        "old_examples": [
            "examples/",  # Old examples (tests migrated, demos should be updated)
        ],
        "duplicate_configs": [
            # After verifying new config system works
            "carla_c2osr/config/duplicate_configs/",  # If exists
        ],
        "migration_artifacts": [
            "scripts/migrate_imports.py",  # Migration script no longer needed
        ],
    }

    return folders_to_clean


def check_folder_status(path: Path):
    """Check if a folder exists and get its size."""
    if not path.exists():
        return None, 0

    if path.is_file():
        return "file", path.stat().st_size

    total_size = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    return "directory", (file_count, total_size)


def format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_cleanup_report(folders_to_clean):
    """Create a detailed report of what will be cleaned."""
    report = []
    total_size = 0
    total_files = 0

    for category, paths in folders_to_clean.items():
        report.append(f"\n## {category.replace('_', ' ').title()}")

        for path_str in paths:
            path = Path(path_str)
            status, info = check_folder_status(path)

            if status is None:
                report.append(f"  - {path_str}: NOT FOUND (already cleaned)")
            elif status == "file":
                report.append(f"  - {path_str}: FILE ({format_size(info)})")
                total_size += info
                total_files += 1
            else:  # directory
                file_count, dir_size = info
                report.append(f"  - {path_str}: DIRECTORY ({file_count} files, {format_size(dir_size)})")
                total_size += dir_size
                total_files += file_count

    report.append(f"\n## Summary")
    report.append(f"Total files to remove: {total_files}")
    report.append(f"Total space to free: {format_size(total_size)}")

    return "\n".join(report), total_files, total_size


def perform_cleanup(folders_to_clean, backup_dir=None, dry_run=False):
    """Perform the actual cleanup."""
    results = {
        "removed": [],
        "backed_up": [],
        "errors": [],
        "not_found": [],
    }

    for category, paths in folders_to_clean.items():
        for path_str in paths:
            path = Path(path_str)

            if not path.exists():
                results["not_found"].append(path_str)
                continue

            try:
                # Create backup if requested
                if backup_dir:
                    backup_path = backup_dir / category / path.name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    if not dry_run:
                        if path.is_file():
                            shutil.copy2(path, backup_path)
                        else:
                            shutil.copytree(path, backup_path)

                    results["backed_up"].append({
                        "original": path_str,
                        "backup": str(backup_path),
                    })

                # Remove the original
                if not dry_run:
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)

                results["removed"].append(path_str)

            except Exception as e:
                results["errors"].append({
                    "path": path_str,
                    "error": str(e),
                })

    return results


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up old directory structure after migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually doing it",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups before removing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report, don't clean anything",
    )

    args = parser.parse_args()

    # Get folders to clean
    folders_to_clean = get_folders_to_clean()

    # Generate report
    report, total_files, total_size = create_cleanup_report(folders_to_clean)

    print("=" * 60)
    print("CLEANUP REPORT - Old Directory Structure")
    print("=" * 60)
    print(report)
    print("=" * 60)

    if args.report_only:
        return

    if total_files == 0:
        print("\nNothing to clean up! All old folders already removed.")
        return

    # Confirm with user unless forced
    if not args.force and not args.dry_run:
        response = input(f"\nProceed with cleanup? This will remove {total_files} files "
                        f"freeing {format_size(total_size)}. (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            return

    # Create backup directory if needed
    backup_dir = None
    if args.backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backups/cleanup_backup_{timestamp}")
        if not args.dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nBackup directory: {backup_dir}")

    # Perform cleanup
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Starting cleanup...")
    results = perform_cleanup(folders_to_clean, backup_dir, args.dry_run)

    # Print results
    print("\n" + "=" * 60)
    print("CLEANUP RESULTS")
    print("=" * 60)

    if results["removed"]:
        print(f"\n✓ Successfully {'would remove' if args.dry_run else 'removed'} {len(results['removed'])} items:")
        for item in results["removed"]:
            print(f"  - {item}")

    if results["backed_up"]:
        print(f"\n✓ {'Would create' if args.dry_run else 'Created'} {len(results['backed_up'])} backups")

    if results["not_found"]:
        print(f"\n○ {len(results['not_found'])} items already cleaned:")
        for item in results["not_found"]:
            print(f"  - {item}")

    if results["errors"]:
        print(f"\n✗ Errors encountered for {len(results['errors'])} items:")
        for error in results["errors"]:
            print(f"  - {error['path']}: {error['error']}")

    # Save results to file
    if not args.dry_run and (results["removed"] or results["errors"]):
        results_file = f"cleanup_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print("\nCleanup complete!" if not args.dry_run else "\nDry run complete!")


if __name__ == "__main__":
    main()