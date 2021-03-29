import pytest


class NotInRange(Exception):
    def __init__(self, message="Value not in range"):
        self.message = message
        super().__init__(self.message)


def test_generic():
    a = 2
    b = 2
    assert a == b
