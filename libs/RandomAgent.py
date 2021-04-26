import numpy as np

from libs.GoGame import GoGame

class RandomAgent():
    def __init__(self):
        pass
    
    def predict(self, game: GoGame):
        actions = game.get_legal_actions()
        return np.random.choice(actions)
