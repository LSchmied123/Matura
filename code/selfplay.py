"""Self-play training loop for Nine Men's Morris.

This module runs self-play games between two instances of the same neural
network, stores experiences in a replay buffer, and trains a policy/Q-network
using a target network for stability. It relies on `rules.py` for game logic
and `NeuralNetData.py` for the model architecture.
"""

import os
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import rules
import NeuralNetData
import NNEvaluation

# Configuration
class SelfPlayConfig:
    # Self-play parameters
    SELFPLAY_GAMES = 100            # Games to play per iteration
    TRAINING_ITERATIONS = 1000      # Number of self-play iterations

    # Neural network parameters
    STATE_SIZE = 28
    ACTION_SIZE = 88
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    EPOCHS_PER_ITERATION = 10       # train steps per iteration

    # Experience replay
    MEMORY_SIZE = 500_000
    MIN_MEMORY_SIZE = 1_000

    # Temperature 
    TEMPERATURE = 0.3               # Higher => more exploration
    TEMPERATURE_DECAY = 0.995
    MIN_TEMPERATURE = 0.1

    # Model saving
    SAVE_INTERVAL = 10              # Save every N iterations
    MODEL_SAVE_DIR = "selfplay_checkpoints_large"
    CHECKPOINT_EPOCH = 0            # For naming only
    CHECKPOINT_PATH = None          # Path to resume from (None => scratch)

    # Evaluation
    EVAL_INTERVAL = 5
    EVAL_GAMES = 50


class ExperienceBuffer:
    """Buffer to store game experiences for training."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)


class SelfPlayTrainer:
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.device = torch.device("cpu")

        # Initialize neural networks
        self.policy_net = NeuralNetData.BiggerPolicyNetwork(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
        ).to(self.device)

        self.target_net = NeuralNetData.BiggerPolicyNetwork(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)

        # Load pre-trained model
        if config.CHECKPOINT_PATH and os.path.exists(config.CHECKPOINT_PATH):
            self.load_checkpoint(config.CHECKPOINT_PATH)
            print(f"Loaded model weights from: {config.CHECKPOINT_PATH}")
        else:
            print("Starting training from scratch")

        # Experience buffer and training stats
        self.experience_buffer = ExperienceBuffer(config.MEMORY_SIZE)
        self.iteration_win_rate = []
        self.iteration_wins = []
        self.temperature = config.TEMPERATURE

        # Ensure save directory exists
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

        self.actions = rules.get_actions()

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(checkpoint)

    def state_to_tensor(self, board, move_count: int, player: int):
        """game state as the 28-dimensional vector.

        Layout (28):
        - 24 values: board at `rules.VALID_POSITIONS` (0 empty, 1 P1, 2 P2)
        - 4 values: [player1_in_hand, player2_in_hand, total_on_board_p1, total_on_board_p2]
        """
        board_positions = [board[r][c] for (r, c) in rules.VALID_POSITIONS]
        p1_pool, p2_pool = rules.count_pieces(board)
        total_p1 = sum(cell == 1 for row in board for cell in row)
        total_p2 = sum(cell == 2 for row in board for cell in row)
        state = board_positions + [p1_pool, p2_pool, total_p1, total_p2]
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, board, move_count: int, player: int, use_exploration: bool = True):
        """Select an action using the policy network with optional exploration.

        Returns (selected_move, action_index) or (None, None) if no moves.
        """
        state_tensor = self.state_to_tensor(board, move_count, player)

        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            action_probs = torch.softmax(logits, dim=1)

        # Get valid moves and map to actions
        valid_moves = rules.get_possible_moves(board, player, move_count)
        if not valid_moves:
            return None, None

        valid_indices = []
        for mv in valid_moves:
            valid_indices.append(self.actions.index(mv))
    
        # remove invalid moves
        masked = torch.zeros_like(action_probs)
        for idx in valid_indices:
            masked[0, idx] = action_probs[0, idx]

        if masked.sum() <= 0:
            # weird error that I managed to trigger
            for idx in valid_indices:
                masked[0, idx] = 1.0 / len(valid_indices)
        else:
            masked = masked / masked.sum()

        # Exploration using temperature
        if use_exploration and self.temperature > self.config.MIN_TEMPERATURE:
            scaled = torch.pow(masked, 1.0 / max(self.temperature, 1e-6))
            scaled = scaled / scaled.sum()
            action_index = torch.multinomial(scaled, 1).item()
        else:
            action_index = torch.argmax(masked, dim=1).item()

        return self.actions[action_index], action_index


    def play_self_game(self):
        board, move_count, _, board_history, game_moves = NNEvaluation.initialize_variables()

        game_states = [] 
        game_actions = [] 

        winner = None
        current_player = 1

        while True:
            state_tensor = self.state_to_tensor(board, move_count, current_player)

            # Select an action
            move, action_index = self.select_action(board, move_count, current_player)
            if move is None:
                # failsafe for no legal moves
                winner = 2 if current_player == 1 else 1
                break

            game_states.append(state_tensor.squeeze(0).detach().cpu().numpy())
            game_actions.append(action_index)

            # Apply move 
            success, new_board = rules.apply_move(copy.deepcopy(board), move, current_player, deterministic=True)
            if not success:
                # failsafe because weird errors
                winner = 2 if current_player == 1 else 1
                break

            board = new_board
            game_moves.append(move)
            board_history.append(rules.board_to_key(board))
            move_count += 1

            # Terminal checks
            if rules.check_game_over(board, move_count):
                winner = current_player
                break
            if board_history.count(rules.board_to_key(board)) >= 3:
                winner = None  # Draw by repetition
                break

            # Switch player
            current_player = 2 if current_player == 1 else 1

        # Assign rewards
        for i, (state_vec, action_idx) in enumerate(zip(game_states, game_actions)):
            player_turn = 1 if i % 2 == 0 else 2
            if winner is None:
                reward = 0.0
            elif winner == player_turn:
                reward = 1.0
            else:
                reward = -1.0

            next_state = game_states[i + 1] if i + 1 < len(game_states) else np.zeros_like(state_vec)
            done = (i == len(game_states) - 1)
            self.experience_buffer.add(state_vec, action_idx, reward, next_state, done)

        return winner, len(game_moves)

    def train_network(self) -> float:
        if self.experience_buffer.size() < self.config.MIN_MEMORY_SIZE:
            return 0.0

        total_loss = 0.0
        gamma = 0.99
        for _ in range(self.config.EPOCHS_PER_ITERATION):
            batch = self.experience_buffer.sample(self.config.BATCH_SIZE)

            # Convert to tensors
            states = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array([e[1] for e in batch]), dtype=torch.long, device=self.device)
            rewards = torch.tensor(np.array([e[2] for e in batch]), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array([e[4] for e in batch]), dtype=torch.bool, device=self.device)

            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q * (~dones)

            loss = nn.MSELoss()(q_values, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.config.EPOCHS_PER_ITERATION

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate_against_random(self):
        """Evaluate current model against a random player."""
        player1wins, player2wins, draws = NNEvaluation.match('2', '4', self.config.EVAL_GAMES, self.policy_net)[2:]
        win_rate = player1wins / self.config.EVAL_GAMES
        return win_rate, player1wins, draws

    def train(self):
        for iteration in range(self.config.TRAINING_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1}/{self.config.TRAINING_ITERATIONS} ---")

            wins = 0
            draws = 0
            total_moves = 0

            # Play self-play games
            for _ in range(self.config.SELFPLAY_GAMES):
                winner, num_moves = self.play_self_game()
                total_moves += num_moves
                if winner is None:
                    draws += 1
                elif winner in (1, 2):
                    wins += 1

            # Train from buffer
            avg_loss = self.train_network()

            # Periodically update target network
            if iteration % 5 == 0:
                self.update_target_network()

            # Decay exploration with temperature
            self.temperature = max(self.config.MIN_TEMPERATURE, self.temperature * self.config.TEMPERATURE_DECAY)

            # Stats
            win_rate = wins / self.config.SELFPLAY_GAMES
            draw_rate = draws / self.config.SELFPLAY_GAMES
            avg_moves = total_moves / self.config.SELFPLAY_GAMES

            self.iteration_win_rate.append(win_rate)
            self.iteration_wins.append(wins)

            print(f"Win rate: {win_rate:.3f}, Draw rate: {draw_rate:.3f}")
            print(f"Average moves per game: {avg_moves:.1f}")
            print(f"Average loss: {avg_loss:.6f}")
            print(f"Temperature: {self.temperature:.3f}")
            print(f"Experience buffer size: {self.experience_buffer.size()}")

            # Save models
            if iteration % self.config.SAVE_INTERVAL == 0:
                model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"selfplay_model_iter_{iteration + self.config.CHECKPOINT_EPOCH}.pth")
                torch.save(self.policy_net.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            # evaluation vs random
            if (iteration + 1) % self.config.EVAL_INTERVAL == 0:
                wr_rand, wins_rand, draws_rand = self.evaluate_against_random()
                losses_rand = self.config.EVAL_GAMES - wins_rand - draws_rand
                print(f"Eval vs Random: Win rate {wr_rand:.3f} ({wins_rand}W/{draws_rand}D/{losses_rand}L)")

        # Final save and plot
        final_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "selfplay_final_model.pth")
        torch.save(self.policy_net.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

        self.plot_training_progress()

    def plot_training_progress(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.iteration_win_rate)
        plt.title('Win Rate Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.iteration_wins)
        plt.title('Wins Per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Wins')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        temps = [self.config.TEMPERATURE * (self.config.TEMPERATURE_DECAY ** i) for i in range(len(self.iteration_win_rate))]
        plt.plot(temps)
        plt.title('Temperature Decay')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.grid(True)

        plt.tight_layout()
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'training_progress.png'))
        plt.show()


def main():
    config = SelfPlayConfig()
    trainer = SelfPlayTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()