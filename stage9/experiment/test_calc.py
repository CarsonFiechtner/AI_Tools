"""Run with: python test_calc.py  (prints OK when both bugs are fixed)."""

from calc import add, average

assert add(2, 3) == 5, f"add(2, 3) should be 5, got {add(2, 3)}"
assert average([1, 2, 3]) == 2, f"average([1, 2, 3]) should be 2, got {average([1, 2, 3])}"
print("OK")
