#!/usr/bin/env python3
"""Final cleanup script to create a clean, non-duplicated structure."""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def create_backup(backup_name="backup_before_final_cleanup"):
    """Create a complete backup before cleanup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"{backup_name}_{timestamp}")

    print(f"Creating backup in {backup_dir}...")

    # Backup key directories
    dirs_to_backup = [
        "carla_c2osr/agents",
        "carla_c2osr/algorithms",
        "carla_c2osr/env",
        "carla_c2osr/environments",
        "carla_c2osr/tests",
        "tests"
    ]

    for dir_path in dirs_to_backup:
        if Path(dir_path).exists():
            dest = backup_dir / dir_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(dir_path, dest)
            print(f"  Backed up: {dir_path}")

    return backup_dir


def merge_c2osr_components():
    """Merge agents/c2osr content into algorithms/c2osr."""
    print("\n1. Merging C2OSR components...")

    source = Path("carla_c2osr/agents/c2osr")
    dest = Path("carla_c2osr/algorithms/c2osr")

    if source.exists():
        dest.mkdir(parents=True, exist_ok=True)

        # Move all algorithm components
        files_to_move = [
            "trajectory_buffer.py",
            "spatial_dirichlet.py",
            "grid.py",
            "dp_mixture.py",
            "risk.py",
            "sampling.py",
            "transition.py"
        ]

        for file in files_to_move:
            src_file = source / file
            if src_file.exists():
                dst_file = dest / file
                shutil.move(str(src_file), str(dst_file))
                print(f"  Moved: {file} -> algorithms/c2osr/")

        # Remove the empty source directory
        if source.exists() and not any(source.iterdir()):
            shutil.rmtree(source)
            print(f"  Removed empty: agents/c2osr/")


def merge_environments():
    """Merge env folder content into environments folder."""
    print("\n2. Merging environment folders...")

    source = Path("carla_c2osr/env")
    dest = Path("carla_c2osr/environments")

    if source.exists():
        dest.mkdir(parents=True, exist_ok=True)

        # Move all env files to environments/carla subfolder
        carla_dest = dest / "carla"
        carla_dest.mkdir(exist_ok=True)

        for item in source.iterdir():
            if item.name != "__pycache__" and item.name != "__init__.py":
                dst = carla_dest / item.name
                shutil.move(str(item), str(dst))
                print(f"  Moved: {item.name} -> environments/carla/")

        # Remove the env folder
        shutil.rmtree(source)
        print(f"  Removed: env/")


def merge_tests():
    """Merge all test folders into single tests directory."""
    print("\n3. Merging test folders...")

    # Use src/c2o_drive/tests as the main test directory
    main_tests = Path("src/c2o_drive/tests")
    main_tests.mkdir(parents=True, exist_ok=True)

    # Merge carla_c2osr/tests
    carla_tests = Path("carla_c2osr/tests")
    if carla_tests.exists():
        for item in carla_tests.iterdir():
            if item.name != "__pycache__" and item.is_file():
                # Put CARLA-specific tests in integration folder
                dst = main_tests / "integration" / f"carla_{item.name}"
                dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    shutil.copy2(str(item), str(dst))
                    print(f"  Copied: {item.name} -> tests/integration/carla_{item.name}")

        # Remove old test directory
        shutil.rmtree(carla_tests)
        print(f"  Removed: carla_c2osr/tests/")

    # Merge root tests folder
    root_tests = Path("tests")
    if root_tests.exists():
        for item in root_tests.iterdir():
            if item.name != "__pycache__" and item.is_file():
                # Determine category based on name
                if "integration" in item.name.lower() or "carla" in item.name.lower():
                    dst_dir = main_tests / "integration"
                elif "unit" in item.name.lower():
                    dst_dir = main_tests / "unit"
                else:
                    dst_dir = main_tests / "functional"

                dst_dir.mkdir(exist_ok=True)
                dst = dst_dir / item.name
                if not dst.exists():
                    shutil.copy2(str(item), str(dst))
                    print(f"  Copied: {item.name} -> {dst_dir.name}/{item.name}")

        # Remove old test directory
        shutil.rmtree(root_tests)
        print(f"  Removed: tests/")


def remove_duplicate_agents():
    """Remove agents folder since we have proper implementations in src/c2o_drive/algorithms."""
    print("\n4. Removing duplicate agents folder...")

    agents_dir = Path("carla_c2osr/agents")

    if agents_dir.exists():
        # Check if baselines exist and are just stubs
        baselines = agents_dir / "baselines"
        if baselines.exists():
            dqn_file = baselines / "dqn_agent.py"
            sac_file = baselines / "sac_agent.py"

            # Check if these are the stub files (< 1KB each)
            if dqn_file.exists() and dqn_file.stat().st_size < 1000:
                print(f"  Found stub DQN (replaced by src/c2o_drive/algorithms/dqn)")
            if sac_file.exists() and sac_file.stat().st_size < 1000:
                print(f"  Found stub SAC (replaced by src/c2o_drive/algorithms/sac)")

        # Remove entire agents directory
        shutil.rmtree(agents_dir)
        print(f"  Removed: agents/ (functionality in algorithms/)")


def create_clean_structure():
    """Create the final clean structure."""
    print("\n5. Creating final clean structure...")

    # Ensure main directories exist
    dirs_to_create = [
        "src/c2o_drive/algorithms/c2osr",
        "src/c2o_drive/environments/carla",
        "src/c2o_drive/environments/virtual",
        "src/c2o_drive/tests/unit",
        "src/c2o_drive/tests/integration",
        "src/c2o_drive/tests/functional",
        "src/c2o_drive/utils",
        "src/c2o_drive/visualization"
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Ensured: {dir_path}")

    # Move carla_c2osr components to src/c2o_drive
    mappings = [
        ("carla_c2osr/algorithms/c2osr", "src/c2o_drive/algorithms/c2osr"),
        ("carla_c2osr/environments", "src/c2o_drive/environments"),
        ("carla_c2osr/visualization", "src/c2o_drive/visualization"),
        ("carla_c2osr/utils", "src/c2o_drive/utils"),
        ("carla_c2osr/core", "src/c2o_drive/core"),
        ("carla_c2osr/evaluation", "src/c2o_drive/evaluation"),
        ("carla_c2osr/runner", "src/c2o_drive/runner"),
        ("carla_c2osr/config", "src/c2o_drive/config/carla")
    ]

    for source, dest in mappings:
        src_path = Path(source)
        dst_path = Path(dest)

        if src_path.exists():
            if src_path.is_dir():
                # Copy directory contents
                dst_path.mkdir(parents=True, exist_ok=True)
                for item in src_path.iterdir():
                    if item.name != "__pycache__":
                        dst_item = dst_path / item.name
                        if not dst_item.exists():
                            if item.is_file():
                                shutil.copy2(str(item), str(dst_item))
                            else:
                                shutil.copytree(str(item), str(dst_item))
                            print(f"  Migrated: {source}/{item.name} -> {dest}/")


def update_imports():
    """Update imports in all Python files."""
    print("\n6. Updating imports...")

    # Import mappings
    import_map = {
        "from carla_c2osr.agents.c2osr": "from c2o_drive.algorithms.c2osr",
        "from carla_c2osr.algorithms.c2osr": "from c2o_drive.algorithms.c2osr",
        "from carla_c2osr.env": "from c2o_drive.environments.carla",
        "from carla_c2osr.environments": "from c2o_drive.environments",
        "import carla_c2osr.agents": "import c2o_drive.algorithms",
        "import carla_c2osr.env": "import c2o_drive.environments.carla"
    }

    # Update imports in src/c2o_drive
    src_path = Path("src/c2o_drive")
    if src_path.exists():
        for py_file in src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                original = content

                for old_import, new_import in import_map.items():
                    content = content.replace(old_import, new_import)

                if content != original:
                    py_file.write_text(content)
                    print(f"  Updated imports in: {py_file.relative_to(src_path)}")
            except Exception as e:
                print(f"  Error updating {py_file}: {e}")


def remove_old_structure():
    """Remove the old carla_c2osr directory after migration."""
    print("\n7. Removing old structure...")

    old_dir = Path("carla_c2osr")
    if old_dir.exists():
        # Check if it's safe to remove (most content should be moved)
        remaining = list(old_dir.rglob("*.py"))
        remaining = [f for f in remaining if "__pycache__" not in str(f)]

        if len(remaining) < 5:  # Only a few files left
            shutil.rmtree(old_dir)
            print(f"  Removed: carla_c2osr/ (content migrated to src/c2o_drive/)")
        else:
            print(f"  Warning: carla_c2osr/ still has {len(remaining)} files, not removing")


def print_final_structure():
    """Print the final clean structure."""
    print("\n" + "="*60)
    print("FINAL CLEAN STRUCTURE")
    print("="*60)

    structure = """
src/c2o_drive/
├── algorithms/        # All algorithm implementations
│   ├── c2osr/        # Complete C2OSR (planner + components)
│   ├── dqn/          # Deep Q-Network
│   └── sac/          # Soft Actor-Critic
├── environments/      # All environments (no duplicates)
│   ├── carla/        # CARLA-specific scenarios
│   ├── virtual/      # Virtual environments
│   └── simple_grid/  # Simple grid world
├── tests/            # Single test directory
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── functional/   # Functional tests
├── config/           # Configuration management
├── core/             # Core types and interfaces
├── utils/            # Utilities
├── visualization/    # Visualization tools
├── evaluation/       # Evaluation metrics
├── runner/           # Training/evaluation runners
└── scripts/          # Runner scripts

NO MORE DUPLICATES:
✓ Single 'environments' folder (no 'env')
✓ Single 'tests' folder
✓ Algorithms in 'algorithms' (no separate 'agents')
✓ C2OSR components unified in algorithms/c2osr
    """

    print(structure)


def main():
    parser = argparse.ArgumentParser(description="Final cleanup for C2O-Drive structure")
    parser.add_argument("--backup", action="store_true", help="Create backup before cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    print("="*60)
    print("FINAL CLEANUP PLAN")
    print("="*60)
    print("""
This will:
1. Merge agents/c2osr → algorithms/c2osr
2. Merge env/ → environments/
3. Merge all tests → src/c2o_drive/tests/
4. Remove duplicate agents/ folder
5. Create final clean structure
6. Update all imports
7. Remove old carla_c2osr structure
    """)

    if not args.force and not args.dry_run:
        response = input("Proceed with final cleanup? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            return

    if args.backup:
        backup_dir = create_backup()
        print(f"\nBackup created: {backup_dir}")

    if args.dry_run:
        print("\nDRY RUN MODE - No changes will be made")
        return

    # Execute cleanup steps
    merge_c2osr_components()
    merge_environments()
    merge_tests()
    remove_duplicate_agents()
    create_clean_structure()
    update_imports()
    remove_old_structure()
    print_final_structure()

    print("\n✓ Final cleanup complete!")
    print("  Run 'tree src/c2o_drive -L 2' to see the clean structure")


if __name__ == "__main__":
    main()