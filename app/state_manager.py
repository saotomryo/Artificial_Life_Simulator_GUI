from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AppState:
    agent_file: Optional[str] = None
    config_files: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "AppState":
        return cls(
            agent_file=data.get("agent_file"),
            config_files=dict(data.get("config_files", {})),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


class StateManager:
    """
    Persists file selections so the application can restore them on startup.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        if state_path is None:
            state_path = Path.cwd() / ".alsim_state.json"
        self.state_path = state_path
        self.state = AppState()
        self._load()

    def _load(self) -> None:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self.state = AppState.from_dict(data)
        except Exception:
            # Corrupted state file; ignore and start fresh.
            self.state = AppState()

    def save(self) -> None:
        try:
            self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception:
            # Non-fatal; ignore writes that fail (e.g., permissions).
            pass

    def set_agent_file(self, path: Path) -> None:
        self.state.agent_file = str(path)
        self.save()

    def get_agent_file(self) -> Optional[Path]:
        if not self.state.agent_file:
            return None
        return Path(self.state.agent_file)

    def set_config_file(self, section: str, path: Path) -> None:
        self.state.config_files[section] = str(path)
        self.save()

    def get_config_file(self, section: str) -> Optional[Path]:
        value = self.state.config_files.get(section)
        if not value:
            return None
        return Path(value)

    def iter_config_files(self):
        for section, path in self.state.config_files.items():
            yield section, Path(path)
