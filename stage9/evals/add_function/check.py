from text_utils import is_palindrome, shout

assert is_palindrome("racecar")
assert is_palindrome("Never odd or even")
assert not is_palindrome("hello")
assert shout("hi") == "HI!", "shout was changed"
