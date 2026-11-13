text = open("greeting.txt", encoding="utf-8").read()
assert text.strip() == "Hello, world!", f"greeting.txt contains {text!r}"
