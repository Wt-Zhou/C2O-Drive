#!/usr/bin/env python3
"""Script to automatically migrate import statements to the new architecture.

Usage:
    python migrate_imports.py <file_or_directory>
    python migrate_imports.py --dry-run <file_or_directory>
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Tuple


# Mapping of old import patterns to new ones
IMPORT_MAPPINGS = {
    # Core types
    r"from carla_c2osr\.env\.types import": "from c2o_drive.core.types import",
    r"import carla_c2osr\.env\.types": "import c2o_drive.core.types",

    # Q-value calculator
    r"from carla_c2osr\.evaluation\.q_value_calculator import": "from c2o_drive.algorithms.c2osr.q_value import",
    r"import carla_c2osr\.evaluation\.q_value_calculator": "import c2o_drive.algorithms.c2osr.q_value",

    # Dirichlet
    r"from carla_c2osr\.agents\.c2osr\.spatial_dirichlet import": "from c2o_drive.algorithms.c2osr.dirichlet import",
    r"import carla_c2osr\.agents\.c2osr\.spatial_dirichlet": "import c2o_drive.algorithms.c2osr.dirichlet",

    # Trajectory buffer
    r"from carla_c2osr\.agents\.c2osr\.trajectory_buffer import": "from c2o_drive.algorithms.c2osr.trajectory_buffer import",
    r"import carla_c2osr\.agents\.c2osr\.trajectory_buffer": "import c2o_drive.algorithms.c2osr.trajectory_buffer",

    # Grid mapper
    r"from carla_c2osr\.agents\.c2osr\.grid import": "from c2o_drive.algorithms.c2osr.grid_mapper import",
    r"import carla_c2osr\.agents\.c2osr\.grid": "import c2o_drive.algorithms.c2osr.grid_mapper",

    # Rewards
    r"from carla_c2osr\.evaluation\.rewards import": "from c2o_drive.algorithms.c2osr.rewards import",
    r"import carla_c2osr\.evaluation\.rewards": "import c2o_drive.algorithms.c2osr.rewards",

    # Collision detector
    r"from carla_c2osr\.evaluation\.collision_detector import": "from c2o_drive.utils.collision import",
    r"import carla_c2osr\.evaluation\.collision_detector": "import c2o_drive.utils.collision",

    # Config
    r"from carla_c2osr\.config import": "from c2o_drive.config import",
    r"import carla_c2osr\.config": "import c2o_drive.config",

    # Algorithm base classes
    r"from carla_c2osr\.algorithms\.base import": "from c2o_drive.algorithms.base import",
    r"import carla_c2osr\.algorithms\.base": "import c2o_drive.algorithms.base",

    # Algorithm config
    r"from carla_c2osr\.algorithms\.c2osr\.config import": "from c2o_drive.algorithms.c2osr.config import",
    r"from carla_c2osr\.algorithms\.c2osr\.internal import": "from c2o_drive.algorithms.c2osr.internal import",

    # Core planner
    r"from carla_c2osr\.core\.planner import": "from c2o_drive.core.planner import",
    r"import carla_c2osr\.core\.planner": "import c2o_drive.core.planner",
}


def migrate_file(filepath: str, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Migrate import statements in a single Python file.

    Args:
        filepath: Path to the Python file
        dry_run: If True, don't write changes, just report them

    Returns:
        Tuple of (number of changes, list of change descriptions)
    """
    if not filepath.endswith('.py'):
        return 0, []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, []

    original_content = content
    changes = []

    for old_pattern, new_replacement in IMPORT_MAPPINGS.items():
        matches = re.findall(old_pattern + r".*", content)
        if matches:
            for match in matches:
                new_line = re.sub(old_pattern, new_replacement, match)
                content = content.replace(match, new_line)
                changes.append(f"  {match.strip()} -> {new_line.strip()}")

    if content != original_content:
        if not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ Updated {filepath} ({len(changes)} changes)")
            except Exception as e:
                print(f"✗ Error writing {filepath}: {e}")
                return 0, []
        else:
            print(f"Would update {filepath} ({len(changes)} changes)")

        return len(changes), changes

    return 0, []


def migrate_directory(directory: str, dry_run: bool = False) -> None:
    """Recursively migrate all Python files in a directory.

    Args:
        directory: Root directory to search
        dry_run: If True, don't write changes, just report them
    """
    total_files = 0
    total_changes = 0
    all_changes = []

    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                changes_count, changes = migrate_file(filepath, dry_run)
                if changes_count > 0:
                    total_files += 1
                    total_changes += changes_count
                    all_changes.extend(changes)

    print(f"\n{'='*60}")
    print(f"Migration Summary {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"Files updated: {total_files}")
    print(f"Total changes: {total_changes}")

    if all_changes and dry_run:
        print(f"\nChanges that would be made:")
        for change in all_changes[:10]:  # Show first 10 changes
            print(change)
        if len(all_changes) > 10:
            print(f"  ... and {len(all_changes) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Migrate import statements to new architecture')
    parser.add_argument('path', help='File or directory to migrate')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist")
        sys.exit(1)

    if os.path.isfile(args.path):
        changes_count, changes = migrate_file(args.path, args.dry_run)
        if changes:
            print("\nChanges made:")
            for change in changes:
                print(change)
    else:
        migrate_directory(args.path, args.dry_run)


if __name__ == "__main__":
    main()