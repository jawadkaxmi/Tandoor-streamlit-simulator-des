# disciplines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from engine import Discipline, Customer


@dataclass
class FCFS(Discipline):
    """First-Come First-Served (FIFO) queue."""
    name: str = "FCFS"

    def reset(self) -> None:
        self._q: List[Customer] = []

    def push(self, customer: Customer) -> None:
        self._q.append(customer)

    def pop(self) -> Customer:
        if not self._q:
            raise IndexError("FCFS queue empty")
        return self._q.pop(0)

    def __len__(self) -> int:
        return len(self._q)
