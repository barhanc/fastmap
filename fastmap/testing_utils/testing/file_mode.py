from enum import Enum


class FileMode(str, Enum):
    READ = 'r'
    WRITE = 'w'
    APPEND = 'a'