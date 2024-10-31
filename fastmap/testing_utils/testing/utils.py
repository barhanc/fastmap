from enum import Enum

from typing import TypeVar


T = TypeVar('T')
V = TypeVar('V')

class Color(str, Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

def success(text: str) -> str:
    """Color text in green."""
    return f"{Color.GREEN}{text}{Color.RESET}"

def fail(text: str) -> str:
    """Color text in red."""
    return f"{Color.RED}{text}{Color.RESET}"