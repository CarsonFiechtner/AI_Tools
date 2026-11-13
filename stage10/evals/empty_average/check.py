from stats import average

assert average([]) == 0, f"average([]) gave {average([])}"
assert average([2, 4]) == 3, "average of a non-empty list changed"
