"""
Demonstration of UUID prefix sharding for raw signal storage.

This script illustrates how UUID-based sharding distributes device data
across hierarchical directories, preventing file system performance issues.

Run this to see sharding in action:
    python scripts/demo_sharding.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.database import Database
from src.database.init_db import generate_device_id


def demo_sharding():
    """Demonstrate UUID prefix sharding with example UUIDs."""
    print("=" * 80)
    print("UUID PREFIX SHARDING DEMONSTRATION")
    print("=" * 80)
    print()

    db = Database(":memory:")

    # Example UUIDs with different prefixes
    example_uuids = [
        ("550e8400-e29b-41d4-a716-446655440000", "Production Sensor Alpha"),
        ("a1b2c3d4-e5f6-7890-abcd-ef1234567890", "Test Device Beta"),
        ("00000000-0000-0000-0000-000000000001", "Development Unit 001"),
        ("ffffffff-ffff-ffff-ffff-ffffffffffff", "Edge Case Max"),
        ("12345678-90ab-cdef-1234-567890abcdef", "Field Device Gamma"),
    ]

    print("📊 SHARDING EXAMPLES:")
    print("-" * 80)
    print(f"{'Device UUID':<40} | {'Shard Path':<35}")
    print("-" * 80)

    for device_id, _name in example_uuids:
        prefix1, prefix2 = db._get_shard_path(device_id)
        shard_path = f"{prefix1}/{prefix2}/{device_id[:8]}.../"
        print(f"{device_id:<40} | {shard_path:<35}")

    print()
    print("=" * 80)
    print("🔢 DISTRIBUTION ANALYSIS (1000 Random UUIDs)")
    print("=" * 80)
    print()

    # Generate 1000 random UUIDs and analyze distribution
    shard_counts = {}
    for _ in range(1000):
        device_id = generate_device_id()
        prefix1, prefix2 = db._get_shard_path(device_id)
        shard_key = f"{prefix1}/{prefix2}"
        shard_counts[shard_key] = shard_counts.get(shard_key, 0) + 1

    unique_shards = len(shard_counts)
    max_per_shard = max(shard_counts.values())
    min_per_shard = min(shard_counts.values())
    avg_per_shard = 1000 / unique_shards

    print("Total UUIDs:              1000")
    print(f"Unique Shards:            {unique_shards} (out of 65,536 possible)")
    print(f"Distribution:             {(unique_shards / 1000) * 100:.1f}% unique")
    print(f"Max in single shard:      {max_per_shard}")
    print(f"Min in single shard:      {min_per_shard}")
    print(f"Average per shard:        {avg_per_shard:.2f}")
    print()

    print("=" * 80)
    print("📁 DIRECTORY STRUCTURE BENEFITS")
    print("=" * 80)
    print()

    print("WITHOUT SHARDING (Flat Structure):")
    print("  data/raw_signals/")
    print("    ├── device-1/")
    print("    ├── device-2/")
    print("    ├── ...")
    print("    └── device-50000/      ⚠️  OS performance degrades >10,000 files/dir")
    print()

    print("WITH 2-LEVEL SHARDING (Hierarchical):")
    print("  data/raw_signals/")
    print("    ├── 00/")
    print("    │   ├── 00/")
    print("    │   │   └── 00000000-.../ (1-2 devices)")
    print("    │   ├── 01/")
    print("    │   │   └── 00010000-.../ (1-2 devices)")
    print("    │   └── ...")
    print("    ├── 55/")
    print("    │   ├── 0e/")
    print("    │   │   └── 550e8400-.../ (1-2 devices)")
    print("    │   └── ...")
    print("    └── ff/")
    print("        └── ff/")
    print("            └── ffffffff-.../ (1-2 devices)")
    print()
    print("  ✅ 65,536 possible shards → avg ~0.76 devices per shard (at 50K scale)")
    print("  ✅ DVC can parallelize across shards")
    print("  ✅ S3/DagsHub list operations use prefix filtering")
    print()

    print("=" * 80)
    print("📈 SCALABILITY METRICS")
    print("=" * 80)
    print()

    scales = [
        (1_000, "Small deployment"),
        (10_000, "Medium scale"),
        (100_000, "Large production"),
        (1_000_000, "Enterprise scale"),
    ]

    print(f"{'Device Count':<20} | {'Avg/Shard':<15} | {'Max/Shard*':<15} | {'Scenario':<20}")
    print("-" * 80)

    for device_count, scenario in scales:
        avg_per_shard = device_count / 65536
        # Assume Poisson distribution for max estimate
        max_estimate = int(avg_per_shard * 3) if avg_per_shard > 1 else 1

        print(f"{device_count:>19,} | {avg_per_shard:>14.2f} | {max_estimate:>14} | {scenario:<20}")

    print()
    print("*Estimated assuming random UUID distribution")
    print()

    print("=" * 80)
    print("✅ CONCLUSION")
    print("=" * 80)
    print()
    print("UUID prefix sharding is ESSENTIAL for production MLOps systems:")
    print("  1. Prevents file system performance degradation")
    print("  2. Enables DVC parallel processing")
    print("  3. Optimizes cloud storage operations")
    print("  4. Supports unlimited horizontal scaling")
    print("  5. Industry standard (Git, Docker, CDNs)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_sharding()
