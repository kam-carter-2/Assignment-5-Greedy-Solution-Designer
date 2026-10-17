
"""
Greedy Algorithms Assignment

Implement three greedy algorithms for delivery optimization:
1. Package Prioritization (Activity Selection)
2. Truck Loading (Fractional Knapsack)
3. Driver Assignment (Interval Scheduling)
"""

import json
import time
import heapq


# ============================================================================
# PART 1: PACKAGE PRIORITIZATION (Activity Selection)
# ============================================================================

def maximize_deliveries(time_windows):
    """
    Schedule the maximum number of non-overlapping deliveries.

    Args:
        time_windows: List of dictionaries containing delivery_id, start, end.

    Returns:
        List of delivery IDs that can be completed.
    """
    sorted_deliveries = sorted(
        time_windows,
        key=lambda delivery: delivery['end']
    )

    selected = []
    last_end = float('-inf')

    for delivery in sorted_deliveries:
        if delivery['start'] >= last_end:
            selected.append(delivery['delivery_id'])
            last_end = delivery['end']

    return selected


# ============================================================================
# PART 2: TRUCK LOADING (Fractional Knapsack)
# ============================================================================

def optimize_truck_load(packages, weight_limit):
    """
    Maximize total package priority within a truck's weight limit.
    Fractions of packages may be loaded.

    Args:
        packages: List of dictionaries with package_id, weight, priority.
        weight_limit: Maximum truck weight.

    Returns:
        Dictionary containing total_priority, total_weight, and packages.
    """
    if weight_limit <= 0:
        return {
            'total_priority': 0.0,
            'total_weight': 0.0,
            'packages': []
        }

    # Sort by priority value per unit of weight.
    sorted_packages = sorted(
        (
            package for package in packages
            if package['weight'] > 0 and package['priority'] > 0
        ),
        key=lambda package: (
            package['priority'] / package['weight']
        ),
        reverse=True
    )

    total_priority = 0.0
    total_weight = 0.0
    loaded_packages = []
    remaining_capacity = weight_limit

    for package in sorted_packages:
        if remaining_capacity <= 0:
            break

        weight = package['weight']
        fraction = min(1.0, remaining_capacity / weight)

        loaded_packages.append({
            'package_id': package['package_id'],
            'fraction': fraction
        })

        total_weight += weight * fraction
        total_priority += package['priority'] * fraction
        remaining_capacity -= weight * fraction

    return {
        'total_priority': total_priority,
        'total_weight': total_weight,
        'packages': loaded_packages
    }


# ============================================================================
# PART 3: DRIVER ASSIGNMENT (Interval Scheduling)
# ============================================================================

def minimize_drivers(deliveries):
    """
    Assign deliveries to the minimum number of drivers.

    Args:
        deliveries: List of dictionaries containing delivery_id, start, end.

    Returns:
        Dictionary with num_drivers and assignments.
    """
    sorted_deliveries = sorted(
        deliveries,
        key=lambda delivery: (
            delivery['start'],
            delivery['end']
        )
    )

    # Heap entries contain (driver's next available time, driver index).
    available_drivers = []
    assignments = []

    for delivery in sorted_deliveries:
        start = delivery['start']
        end = delivery['end']

        # Reuse the driver who becomes available earliest.
        if available_drivers and available_drivers[0][0] <= start:
            _, driver_index = heapq.heappop(available_drivers)
        else:
            driver_index = len(assignments)
            assignments.append([])

        assignments[driver_index].append(delivery['delivery_id'])

        heapq.heappush(
            available_drivers,
            (end, driver_index)
        )

    return {
        'num_drivers': len(assignments),
        'assignments': assignments
    }


# ============================================================================
# TESTING AND BENCHMARKING
# ============================================================================

def load_scenario(filename):
    """Load a scenario from the scenarios directory."""
    with open(f"scenarios/{filename}", "r") as file:
        return json.load(file)


def test_package_prioritization():
    """Test package prioritization with known cases."""
    print("=" * 70)
    print("TESTING PACKAGE PRIORITIZATION")
    print("=" * 70)

    test1 = [
        {'delivery_id': 'A', 'start': 1, 'end': 2},
        {'delivery_id': 'B', 'start': 3, 'end': 4},
        {'delivery_id': 'C', 'start': 5, 'end': 6}
    ]

    result1 = maximize_deliveries(test1)
    print("\nTest 1: Non-overlapping deliveries")
    print(f"Expected: 3 deliveries | Got: {len(result1)}")
    print("PASS" if len(result1) == 3 else "FAIL")

    test2 = [
        {'delivery_id': 'A', 'start': 1, 'end': 5},
        {'delivery_id': 'B', 'start': 2, 'end': 4},
        {'delivery_id': 'C', 'start': 3, 'end': 6}
    ]

    result2 = maximize_deliveries(test2)
    print("\nTest 2: Overlapping deliveries")
    print(f"Selected: {result2}")
    print(f"Number selected: {len(result2)}")

    test3 = [
        {'delivery_id': 'A', 'start': 1, 'end': 3},
        {'delivery_id': 'B', 'start': 2, 'end': 5},
        {'delivery_id': 'C', 'start': 4, 'end': 7},
        {'delivery_id': 'D', 'start': 6, 'end': 9}
    ]

    result3 = maximize_deliveries(test3)
    print("\nTest 3: Mixed overlapping deliveries")
    print(f"Expected: 2 deliveries | Got: {len(result3)}")
    print(f"Selected: {result3}")
    print("PASS" if len(result3) == 2 else "FAIL")


def test_truck_loading():
    """Test fractional knapsack truck loading."""
    print("\n" + "=" * 70)
    print("TESTING TRUCK LOADING")
    print("=" * 70)

    packages1 = [
        {'package_id': 'A', 'weight': 10, 'priority': 60},
        {'package_id': 'B', 'weight': 20, 'priority': 100}
    ]

    result1 = optimize_truck_load(packages1, 50)
    print("\nTest 1: All packages fit")
    print(f"Expected priority: 160 | Got: {result1['total_priority']}")
    print(f"Total weight: {result1['total_weight']}")
    print(
        "PASS"
        if abs(result1['total_priority'] - 160) < 0.0001
        else "FAIL"
    )

    packages2 = [
        {'package_id': 'A', 'weight': 10, 'priority': 60},
        {'package_id': 'B', 'weight': 20, 'priority': 100},
        {'package_id': 'C', 'weight': 30, 'priority': 120}
    ]

    result2 = optimize_truck_load(packages2, 50)
    print("\nTest 2: Fractional knapsack")
    print(f"Expected priority: 240 | Got: {result2['total_priority']}")
    print(f"Total weight: {result2['total_weight']}")
    print(f"Packages loaded: {result2['packages']}")
    print(
        "PASS"
        if abs(result2['total_priority'] - 240) < 0.0001
        else "FAIL"
    )


def test_driver_assignment():
    """Test minimum-driver assignment."""
    print("\n" + "=" * 70)
    print("TESTING DRIVER ASSIGNMENT")
    print("=" * 70)

    deliveries1 = [
        {'delivery_id': 'A', 'start': 1, 'end': 2},
        {'delivery_id': 'B', 'start': 3, 'end': 4},
        {'delivery_id': 'C', 'start': 5, 'end': 6}
    ]

    result1 = minimize_drivers(deliveries1)
    print("\nTest 1: Non-overlapping deliveries")
    print(f"Expected: 1 driver | Got: {result1['num_drivers']}")
    print("PASS" if result1['num_drivers'] == 1 else "FAIL")

    deliveries2 = [
        {'delivery_id': 'A', 'start': 1, 'end': 5},
        {'delivery_id': 'B', 'start': 2, 'end': 6},
        {'delivery_id': 'C', 'start': 3, 'end': 7}
    ]

    result2 = minimize_drivers(deliveries2)
    print("\nTest 2: All overlapping")
    print(f"Expected: 3 drivers | Got: {result2['num_drivers']}")
    print("PASS" if result2['num_drivers'] == 3 else "FAIL")

    deliveries3 = [
        {'delivery_id': 'A', 'start': 1, 'end': 3},
        {'delivery_id': 'B', 'start': 2, 'end': 4},
        {'delivery_id': 'C', 'start': 4, 'end': 6},
        {'delivery_id': 'D', 'start': 5, 'end': 7}
    ]

    result3 = minimize_drivers(deliveries3)
    print("\nTest 3: Mixed overlapping deliveries")
    print(f"Expected: 2 drivers | Got: {result3['num_drivers']}")
    print(f"Assignments: {result3['assignments']}")
    print("PASS" if result3['num_drivers'] == 2 else "FAIL")


def benchmark_scenarios():
    """Benchmark all algorithms on the provided realistic scenarios."""
    print("\n" + "=" * 70)
    print("BENCHMARKING ON REALISTIC SCENARIOS")
    print("=" * 70)

    print("\nScenario 1: Package Prioritization")
    data = load_scenario("package_prioritization.json")

    start_time = time.perf_counter()
    result = maximize_deliveries(data)
    elapsed = time.perf_counter() - start_time

    print(f"Deliveries scheduled: {len(result)} out of {len(data)}")
    print(f"Runtime: {elapsed * 1000:.4f} ms")

    print("\nScenario 2: Truck Loading")
    data = load_scenario("truck_loading.json")

    start_time = time.perf_counter()
    result = optimize_truck_load(
        data['packages'],
        data['truck_capacity']
    )
    elapsed = time.perf_counter() - start_time

    print(f"Total priority: {result['total_priority']:.2f}")
    print(f"Total weight: {result['total_weight']:.2f} lbs")
    print(f"Packages used: {len(result['packages'])}")
    print(f"Runtime: {elapsed * 1000:.4f} ms")

    print("\nScenario 3: Driver Assignment")
    data = load_scenario("driver_assignment.json")

    start_time = time.perf_counter()
    result = minimize_drivers(data)
    elapsed = time.perf_counter() - start_time

    print(f"Drivers needed: {result['num_drivers']}")
    print(f"Runtime: {elapsed * 1000:.4f} ms")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("GREEDY ALGORITHMS ASSIGNMENT")
    print("Running all tests and performance benchmarks...\n")

    test_package_prioritization()
    test_truck_loading()
    test_driver_assignment()

    try:
        benchmark_scenarios()
    except FileNotFoundError as error:
        print(f"\nBenchmark files not found: {error}")
        print("Run the delivery generator to create the scenarios first.")
