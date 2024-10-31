from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeGraphProps:
    save_path: str = ""
    name: str = ""
    xlabel: str = ""
    ylabel: str = ""
    steps: list[str] = field(default_factory=list)