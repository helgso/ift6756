import copy
import math
from typing import List

import numpy as np

from libs.Network import Network
from libs.GoGame import GoGame

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0  # N(s, a) is the visit count
        self.to_play = -1
        self.prior = prior  # P(s, a) is the prior probability of selecting a in s
        self.value_sum = 0  # W(s, a) is the total action-value
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        # Q(s, a) = W(s,a)/N(s,a)
        return self.value_sum / self.visit_count
    
    def get_policy(self, game: GoGame):
        sum_visits = sum(child.visit_count for child in self.children.values())
        return [
            self.children[a].visit_count / sum_visits if a in self.children else 0
            for a in range(game.get_action_size())
        ]

class MCTS():
    @staticmethod
    def run(network: Network, game: GoGame, config: dict):
        root = Node(0)
        MCTS.evaluate_and_expand_node(root, game, network)
        MCTS.add_exploration_noise(config, root)

        for _ in range(config['n_simulations']):            
            node = root
            scratch_game = copy.deepcopy(game)
            search_path = [node]

            while node.expanded():
                action, node = MCTS.select_child(config, node)
                scratch_game.step(action)
                search_path.append(node)
            
            value = MCTS.evaluate_and_expand_node(node, scratch_game, network)
            MCTS.backpropagate(search_path, value, scratch_game.to_play())
        
        action, q_value = MCTS.select_action(config, game, root)
        target_policy = root.get_policy(game)
        
        return action, q_value, target_policy
    
    @staticmethod
    def select_action(config: dict, game: GoGame, root: Node):
        action_stats = np.array([[child.visit_count, action, child.value()] for action, child in root.children.items()])
        if len(game.get_history()) < 2*config['n_sampling_moves']:
            return MCTS.softmax_sample(action_stats, config['temperature'])
        
        return MCTS.max(action_stats)

    @staticmethod
    def softmax_sample(action_stats: np.ndarray, temperature: float):
        exps = np.exp(action_stats[:,0,]/temperature)
        probabilities = exps / np.sum(exps)
        choice = np.random.choice(np.arange(len(exps)), p=probabilities)
        return int(action_stats[choice][1]), action_stats[choice][2]
    
    @staticmethod
    def max(action_stats: np.ndarray):
        choice = np.argmax(action_stats[:,0])
        return int(action_stats[choice][1]), action_stats[choice][2]

    @staticmethod
    # Select the child with the highest UCB score
    def select_child(config: dict, node: Node):
        _, action, child = max((MCTS.ucb_score(config, node, child), action, child) for action, child in node.children.items())
        return action, child
    
    # The score for a node is based on its value Q(s, a), plus an exploration bonus U(s, a) based on the prior P(s, a)
    #   -> at = argmax(Q(s, a) + U(s, a))
    #   -> U(s, a) = C(s) * P(s, a) * sqrt(N(s)/(1 + N(s, a)), where:
    #      - N(s) is the parent visit count
    #      - C(s) is the exploration rate: C(s) = log ((1 + N(s) + c_base)/c_base) + c_init,
    @staticmethod
    def ucb_score(config: dict, parent: Node, child: Node):
        c_s = math.log((1 + parent.visit_count + config['pb_c_base']) / config['pb_c_base']) + config['pb_c_init']
        u_s = c_s * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        q_s = child.value()
        return q_s + u_s

    @staticmethod
    def evaluate_and_expand_node(node: Node, game: GoGame, network: Network):
        def expand_node(policy_logits: np.ndarray):
            node.to_play = game.to_play()
            policy = {action: np.exp(policy_logits[action]) for action in game.get_legal_actions()}
            policy_sum = sum(policy.values())
            for action, logit in policy.items():
                node.children[action] = Node(logit / policy_sum)
        policy_logits, value = network.predict(game.get_board_state())
        expand_node(policy_logits[0])
        return value[0][0]

    # Propagate the value up to the root
    @staticmethod
    def backpropagate(search_path: List[Node], value: float, to_play):
        for node in search_path:
            # W(s, a) = W(s, a) + v
            node.value_sum += value if node.to_play == to_play else (1 - value)
            # N(s, a) = N(st, at) + 1
            node.visit_count += 1
            # Q(s, a) = W(st,at) / N(st,at)
    
    # Add dirichlet noise to the prior of the root to encourage the search to explore new actions
    @staticmethod
    def add_exploration_noise(config: dict, node: Node):
        actions = node.children.keys()
        noise = np.random.gamma(config['root_dirichlet_alpha'], 1, len(actions))
        fraction = config['root_exploration_fraction']
        for action, n in zip(actions, noise):
            node.children[action].prior = node.children[action].prior * (1 - fraction) + n * fraction
