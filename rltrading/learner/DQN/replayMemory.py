from collections import deque, namedtuple
import random

Transition = namedtuple(
    "Transition", field_names=["state", "action", "next_state", "reward"]
)


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
