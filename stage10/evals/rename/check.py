import pathlib

for path in pathlib.Path(".").glob("*.py"):
    assert "calc_total" not in path.read_text(), f"calc_total is still in {path}"
from main import receipt

assert receipt([10, 20]) == "Total: 33.0", receipt([10, 20])
