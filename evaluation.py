from typing import List

from tqdm import tqdm
import numpy as np
import elopy

from libs.GoGame import GoGame
from libs.Network import Network
from libs.MCTS import MCTS
from libs.RandomAgent import RandomAgent

def main():
    game_config = {
        'board_size': 9
    }
    network_config = {
        'board_size': game_config['board_size'],
        'kernel_size': 3,
        'n_filters': 128,
        'learning_rate': 1e-5,
        'weight_decay': 1e-3,
        'value_head_dense_layer_size': 128,
        'head_inputs_fixed': True,
        'n_middle_blocks': 0
    }
    self_play_config = {
        'n_games': 30,
        'n_simulations': 150,
        'max_moves': 30,
        
        # First 15 moves (30 ply) are generated with temperature randomness. To quote the paper:
        # "When we forced AlphaZero to play with greater diversity (by softmax sampling
        # with a temperature of 10.0 among moves for which the value was no more than 1% away
        # from the best move for the first 30 plies) the winning rate increased from 5.8% to 14%."
        'n_sampling_moves': 15,
        'temperature': 10,

        'root_dirichlet_alpha': 0.03,
        'root_exploration_fraction': 0.25,

        'pb_c_base': 19652,
        'pb_c_init': 1.25,
    }
    
    game = GoGame(game_config['board_size'])
    
    random_agent = RandomAgent()
    network = Network(config=network_config)
    # Choose a model saved by training.py
    network.load_weights('checkpoints/TargetMode.Z_VALUES/4f468474f5c44c8f8107568ba64f047d/10.h5')

    compete(network, random_agent, game, self_play_config)

def compete(network: Network, random_agent: RandomAgent, game: GoGame, config: dict):
    def alpha_zero_plays():
        action, q_value, target_policy = MCTS.run(network, game, config)
        game.step(action)
    
    def random_plays():
        action = random_agent.predict(game)
        game.step(action)
    
    def get_winner(players: List, black_player: int, reward: float):
        if reward == 0: # Tie
            return None
        elif reward == 1: # Black won
            return players[black_player]['name']
        elif reward == -1: # White won
            return players[1 - black_player]['name']
    
    elo_manager = elopy.Implementation()
    players = [
        {'name': 'AlphaZero', 'play_function': alpha_zero_plays},
        {'name': 'RandomAgent', 'play_function': random_plays}
    ]
    for player in players:
        elo_manager.addPlayer(player['name'])

    for i in tqdm(range(config['n_games']), desc="Playing competition games"):
        game.reset()
        black_player = np.random.choice(len(players))
        white_player = 1 - black_player
        while True:
            players[black_player]['play_function']()
            if game.is_done() or len(game.history) >= config['max_moves']:
                break
            players[white_player]['play_function']()
            if game.is_done() or len(game.history) >= config['max_moves']:
                break
        if not game.is_done():
            game.end_game()
        reward = game.get_reward()
        winner = get_winner(players, black_player, reward)
        if winner is None:
            elo_manager.recordMatch(players[0]['name'], players[1]['name'], draw=True)
        else:
            elo_manager.recordMatch(players[0]['name'], players[1]['name'], winner=winner)
        print(f'\tGame result: {winner + " won" if winner else "Tie"}')
        print(f'\tCurrent ELO ratings: {elo_manager.getRatingList()}')

if __name__ == '__main__':
    main()
