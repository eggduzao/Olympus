
from typing import List, Optional


def binary_search(arr: List[int], target: int) -> Optional[int]:
    """
    Perform binary search on a sorted list of integers.

    Parameters
    ----------
    arr : List[int]
        A sorted list of integers (ascending order).
    target : int
        The integer value to search for.

    Returns
    -------
    Optional[int]
        The index of the target in the list if found, else None.

    Notes
    -----
    - Time complexity: O(log n)
    - Space complexity: O(1)
    - Works with very large arrays (streaming input can be chunked beforehand).
    - Edge cases:
        * Empty list → returns None
        * Duplicates → returns index of *some* occurrence (not guaranteed first/last)
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None


def load_array_from_file(path: str) -> List[int]:
    """
    Load integers (one per line) from a text file and return a sorted list.

    Parameters
    ----------
    path : str
        Path to the input file containing integers.

    Returns
    -------
    List[int]
        Sorted list of integers from file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = [int(line.strip()) for line in f if line.strip()]
    return sorted(data)


if __name__ == "__main__":
    # Example usage with the provided file structure
    filename = "input.txt"
    arr = load_array_from_file(filename)

    print(f"Loaded array (len={len(arr)}): {arr}")

    # Example queries
    for target in [1, 9, 15, 2]:
        idx = binary_search(arr, target)
        if idx is not None:
            print(f"Target {target} found at index {idx}.")
        else:
            print(f"Target {target} not found.")
