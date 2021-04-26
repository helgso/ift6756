import os
import uuid
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from libs.GoGame import GoGame
from libs.Network import Network
from libs.MCTS import MCTS
from libs.shared import TargetMode

def main():    
    target_mode = TargetMode.Q_VALUES

    model_id = uuid.uuid4().hex
    model_folder = f'checkpoints/{target_mode}/{model_id}'

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    target_mode = TargetMode.Q_VALUES
    game_config = {
        'board_size': 9
    }
    network_config = {
        'board_size': game_config['board_size'],
        'kernel_size': 3,
        'n_filters': 128,
        'learning_rate': 1e-5,
        'weight_decay': 1e-3,
        'value_head_dense_layer_size': 128
    }
    train_config = {
        'n_training_loops': 1000,
        'checkpoint_interval': 2,
        'epochs': 1,
        'batch_size': 8
    }
    self_play_config = {
        'n_games': 1,
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

        'target_mode': target_mode
    }

    game = GoGame(game_config['board_size'])
    network = Network(config=network_config)

    print(f'Model id: {model_id}')
    for loop in range(train_config['n_training_loops']):
        if loop % train_config['checkpoint_interval'] == 0:
            network.save_weights(f'{model_folder}/{loop}')
        samples = self_play(network, game, self_play_config)
        train(network, samples, train_config)

def self_play(network: Network, game: GoGame, config: dict):
    samples = []
    for game_i in range(config['n_games']):
        game.reset()
        game_stats = play_game(network, game, config)
        samples += game_stats
    return np.array(samples)

def play_game(network: Network, game: GoGame, config: dict):
    board_states = []
    target_policies = []
    q_values = []
    n_ply = 2*config['max_moves']
    with tqdm(total=n_ply, desc='Playing game with MCTS') as progress_bar:
        while not game.is_done() and len(game.history) < n_ply:
            board_state = game.get_board_state()
            action, q_value, target_policy = MCTS.run(network, game, config)
            game.step(action)
            # Store results
            board_states.append(board_state.tolist())
            target_policies.append(target_policy)
            q_values.append([q_value])
            progress_bar.update(1)
    if not game.is_done():
        game.end_game()
    
    target_values = game.get_z_values().tolist() if config['target_mode'] == TargetMode.Z_VALUES else q_values
    return list(zip(board_states, target_policies, target_values))

def train(network: Network, samples: np.ndarray, config: dict):
    # samples doesn't define each np.ndarray's sub-shape (due to multiple shapes). Let's fix that
    board_states = np.array(samples[:,0].tolist())
    target_policies = np.array(samples[:,1].tolist())
    target_values = np.array(samples[:,2].tolist())

    network.fit(x=board_states, y=(target_policies, target_values),
                batch_size=config['batch_size'], epochs=config['epochs'])

if __name__ == '__main__':
    main()
