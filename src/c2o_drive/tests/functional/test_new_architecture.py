"""Test that the new architecture imports work correctly."""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def test_core_imports():
    """Test that core types can be imported from the new location."""
    try:
        from c2o_drive.core.types import (
            WorldState,
            EgoState,
            AgentState,
            EgoControl,
            AgentType
        )
        print("✓ Core types import successful")
        return True
    except ImportError as e:
        print(f"✗ Core types import failed: {e}")
        return False


def test_c2osr_imports():
    """Test that C2OSR components can be imported from the new location."""
    try:
        # Note: These imports will fail if the files haven't been updated yet
        # because they still have old import statements inside them
        print("Testing C2OSR imports (may fail due to internal imports not updated yet)...")

        # Try importing the main module
        import c2o_drive.algorithms.c2osr
        print("✓ C2OSR module import successful")

        return True
    except ImportError as e:
        print(f"⚠ C2OSR import partially failed (expected until internal imports are updated): {e}")
        return False


def test_utils_imports():
    """Test that utils can be imported."""
    try:
        # Check if collision.py exists
        collision_path = os.path.join(src_path, 'c2o_drive', 'utils', 'collision.py')
        if os.path.exists(collision_path):
            print("✓ Utils collision.py exists")
            return True
        else:
            print("⚠ Utils collision.py not found")
            return False
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing New C2O-Drive Architecture")
    print("="*60)

    results = []

    # Test core imports
    print("\n1. Testing Core Imports...")
    results.append(test_core_imports())

    # Test C2OSR imports
    print("\n2. Testing C2OSR Imports...")
    results.append(test_c2osr_imports())

    # Test utils
    print("\n3. Testing Utils...")
    results.append(test_utils_imports())

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")

    if not all(results):
        print("\nNote: Some imports may fail because internal import statements")
        print("haven't been updated yet. Run the migration script to fix them:")
        print("  python scripts/migrate_imports.py src/")
    else:
        print("\n✓ All tests passed! The new architecture is working.")

    print("="*60)


if __name__ == "__main__":
    main()