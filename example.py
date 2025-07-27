#!/usr/bin/env python3
"""
Example usage of UltraLogLog Python bindings
"""

import ultraloglog


def main():
    print("UltraLogLog Python Binding Example")
    print("=" * 40)

    # Create a new UltraLogLog sketch with precision 12
    ull = ultraloglog.PyUltraLogLog(12)
    print(f"Created UltraLogLog sketch with precision {ull.get_p()}")
    print(f"Initial estimate: {ull.count()}")

    # Add some unique values
    print("\nAdding unique values...")
    unique_values = ["apple", "banana", "cherry", "date", "elderberry"]
    for value in unique_values:
        ull.add_str(value)
        print(f"Added '{value}', estimate: {ull.count():.2f}")

    # Add some duplicates (should not significantly change the estimate)
    print("\nAdding duplicate values...")
    duplicates = ["apple", "banana", "apple"]
    for value in duplicates:
        ull.add_str(value)
        print(f"Added '{value}' (duplicate), estimate: {ull.count():.2f}")

    # Add numeric values
    print("\nAdding numeric values...")
    numbers = [1, 2, 3, 42, 100]
    for num in numbers:
        ull.add_int(num)
        print(f"Added {num}, estimate: {ull.count():.2f}")

    # Add float values
    print("\nAdding float values...")
    floats = [3.14, 2.71, 1.41]
    for f in floats:
        ull.add_float(f)
        print(f"Added {f}, estimate: {ull.count():.2f}")

    print(f"\nFinal estimate: {ull.count():.2f}")
    print(
        f"Actual unique count: {len(set(unique_values + [str(n) for n in numbers] + [str(f) for f in floats]))}"
    )

    # Test merging
    print("\nTesting sketch merging...")
    ull2 = ultraloglog.PyUltraLogLog(12)
    for i in range(1000, 1010):
        ull2.add_int(i)

    print(f"Second sketch estimate: {ull2.count():.2f}")

    merged = ultraloglog.PyUltraLogLog.merge_sketches(ull, ull2)
    print(f"Merged sketch estimate: {merged.count():.2f}")

    # Test downsizing
    print("\nTesting downsizing...")
    downsized = ull.downsize(10)
    print(
        f"Downsized to precision {downsized.get_p()}, estimate: {downsized.count():.2f}"
    )


if __name__ == "__main__":
    main()
