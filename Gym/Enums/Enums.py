from enum import Enum


class EnvType(Enum):
    CartPole = 0
    MountainCarContinuous = 1


class AgentType(Enum):
    DQN = 0
    CTDL = 1
    A2C = 2
    CTDL_A2C = 3