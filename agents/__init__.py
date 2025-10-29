# Agents package initialization
from .base import Agent, MAX_REWARDS
from .oracle import OracleAgent
from .random_agent import RandomAgent
from .q_learning import QLearningAgent, ContinuousQLearningAgent, HarmonicQAgent
from .r_learning import RLAgent, ContinuousRLAgent, MyopicRLearn
from .smart_r import SMARTRLAgent, StateSMARTRLAgent, AdaptiveSMARTRLAgent
from .harmonic_r import HarmonicRLAgent, HarmonicRLAgent2, AdaptiveHarmonicRLAgent, AdaptiveHarmonicRLAgent2
from .bandits import MAB, ContinuesMAB, UCB, ContinuosUCB

__all__ = [
    'Agent', 'MAX_REWARDS', 'OracleAgent', 'RandomAgent', 'QLearningAgent', 'ContinuousQLearningAgent', 'HarmonicQAgent',
    'RLAgent', 'ContinuousRLAgent', 'MyopicRLearn', 'SMARTRLAgent', 'StateSMARTRLAgent', 'AdaptiveSMARTRLAgent',
    'HarmonicRLAgent', 'HarmonicRLAgent2', 'AdaptiveHarmonicRLAgent', 'AdaptiveHarmonicRLAgent2', 'MAB', 'ContinuesMAB', 'UCB', 'ContinuosUCB'
]

