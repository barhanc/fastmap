from dataclasses import dataclass

from .file_mode import FileMode


@dataclass(frozen=True)
class CSVProps:
    file_path: str
    file_mode: FileMode = FileMode.WRITE
    delimiter: str = ','
