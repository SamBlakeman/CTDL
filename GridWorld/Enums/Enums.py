from enum import Enum

class MazeType(Enum):
    random = 1
    direct = 2
    obstacle1 = 3
    obstacle2 = 4

class AgentType(Enum):
    CTDL = 1
    DQN = 2
