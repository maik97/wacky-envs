from dataclasses import dataclass

@dataclass
class ObjectGroup:

    members: list = None

    def __init__(self, members: list):
        self.members = members

    def __call__(self, idx):
        return self.members[idx]
