import os
import ast
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import rules
from torch.utils.data import TensorDataset, DataLoader
import time

# Configuration
class TrainingConfig:
    def __init__(self):
        self.EPOCHS = 10000
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5
        self.GRAD_ACCUM_STEPS = 1       # >1 simulates small batch size
        self.SAVE_EVERY = 10            # save model every N epochs
        self.DATASET_PATH = r"D:\Matura\NNnewer\sepember\minimax_training_set_symmetries.txt"
        self.MODEL_SAVE_DIR = "checkpoints_batch512_s"
        self.LOG_INTERVAL = 1000        # steps between console/pygame updates
        self.MAX_LINES = 500000         # max lines to read from dataset (None=all)
        self.RESTART_EPOCH = 5880       # epoch to resume training (None=from scratch)
        self.BATCH_SIZE = 1024

# Model
class BiggerPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(BiggerPolicyNetwork, self).__init__()
        '''
        self.network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, action_size)
        )
        '''
        self.network = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

# Data Loading
class GameDataLoader:
    def __init__(self, config):
        self.config = config
        self.actions = rules.get_actions()

    def load_dataset(self):
        """Load and preprocess dataset"""
        print(f"Loading dataset from {self.config.DATASET_PATH}")

        states = []
        labels = []

        with open(self.config.DATASET_PATH, "r") as f:
            for i, line in enumerate(f):
                if self.config.MAX_LINES and i >= self.config.MAX_LINES:
                    break
                if not line.strip():
                    continue

                state, move = ast.literal_eval(line.strip())

                # Convert state to tensor
                state_tensor = torch.FloatTensor(state)

                # Convert move to action index
                if isinstance(move, (tuple, list)):
                    action_index = self.actions.index(move)
                else:
                    raise ValueError(f"Unexpected move format: {move}")

                states.append(state_tensor)
                labels.append(action_index)

        print(f"Loaded {len(states)} samples")
        return states, labels

    def create_dataloader(self, states, labels):
        """Create PyTorch DataLoader"""
        dataset = TensorDataset(torch.stack(states), torch.LongTensor(labels))

        if self.config.BATCH_SIZE > 1:
            return DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        else:
            return list(zip(torch.stack(states), torch.LongTensor(labels)))

# Training Utils
class TrainingUtils:
    def accuracy_topk(output, target, topk=(1,)):
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item())
        return res

    def format_seconds(seconds: int) -> str:
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:  # show 0s if nothing else
            parts.append(f"{seconds}s")

        return " ".join(parts)

# Pygame Monitor
class TrainingMonitor:
    def __init__(self):
        pygame.init()
        self.width, self.height = (800, 600)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Training Monitor")

        self.font = pygame.font.SysFont("Arial", 16)
        self.loss_history = []
        self.acc_history = []

    def update(self, step, loss, acc, state=None, target=None, pred=None):
        self.loss_history.append(loss)
        self.acc_history.append(acc)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # clear
        self.screen.fill((30, 30, 30))

        # plot loss
        if len(self.loss_history) > 1:
            points = []
            for i, loss in enumerate(self.loss_history[-200:]):  # last 200 points
                x = i * (self.width // 2) // min(200, len(self.loss_history))
                y = int(self.height // 2 - loss * 100)  # scale factor
                y = max(0, min(self.height // 2, y))
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, (255, 100, 100), False, points, 2)

        # plot accuracy
        if len(self.acc_history) > 1:
            points = []
            for i, acc in enumerate(self.acc_history[-200:]):
                x = i * (self.width // 2) // min(200, len(self.acc_history))
                y = int(self.height // 2 + (1 - acc) * 100)  # scale factor
                y = max(self.height // 2, min(self.height, y))
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, (100, 255, 100), False, points, 2)

        # text info
        info_lines = [
            f"Step: {step}",
            f"Loss: {loss:.4f}",
            f"Acc: {acc:.2%}",
        ]

        if target is not None and pred is not None:
            info_lines.extend([
                f"Target: {target}",
                f"Pred: {pred}",
                f"Correct: {target == pred}"
            ])

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.width // 2 + 10, 10 + i * 20))

        pygame.display.flip()

# Main Trainer Class
class NeuralNetworkTrainer:
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize components
        self.data_loader = GameDataLoader(self.config)
        self.utils = TrainingUtils()
        self.actions = rules.get_actions()

        # Create model save directory
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)

        # Initialize model
        self.model = BiggerPolicyNetwork(28, len(self.actions)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Load checkpoint if specified
        self.start_epoch = self._load_checkpoint()

        # Initialize monitor
        self.monitor = TrainingMonitor()

    def _load_checkpoint(self):
        """Load model checkpoint if specified"""
        if self.config.RESTART_EPOCH is not None:
            checkpoint_path = os.path.join(
                self.config.MODEL_SAVE_DIR,
                f"model_epoch{self.config.RESTART_EPOCH}.pth"
            )
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"Resumed training from checkpoint: {checkpoint_path}")
                return self.config.RESTART_EPOCH
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
        return 0

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        path = os.path.join(self.config.MODEL_SAVE_DIR, f"model_epoch{epoch + self.start_epoch}.pth")
        torch.save(self.model.state_dict(), path)
        return path

    def _train_single_batch(self, state, label, step):
        """Train on a single sample"""
        state = state.to(self.device)
        label = label.to(self.device)

        outputs = self.model(state.unsqueeze(0))
        loss = self.criterion(outputs, label.unsqueeze(0))
        loss.backward()

        # Accumulate gradients
        if (step + 1) % self.config.GRAD_ACCUM_STEPS == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Calculate accuracy
        acc1, = self.utils.accuracy_topk(outputs, label.unsqueeze(0), topk=(1,))

        return loss.item(), acc1, outputs, label

    def _train_batch(self, state_batch, label_batch):
        """Train on a batch of samples"""
        state_batch = state_batch.to(self.device)
        label_batch = label_batch.to(self.device)

        outputs = self.model(state_batch)
        loss = self.criterion(outputs, label_batch)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Calculate accuracy
        acc1, = self.utils.accuracy_topk(outputs, label_batch, topk=(1,))

        return loss.item(), acc1, outputs, label_batch

    def train(self):
        """Main training loop"""
        print("Starting training...")
        starttime = time.time()

        # Load dataset
        states, labels = self.data_loader.load_dataset()

        if self.config.BATCH_SIZE > 1:
            dataloader = self.data_loader.create_dataloader(states, labels)
        else:
            dataset = self.data_loader.create_dataloader(states, labels)

        step = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            running_loss, running_acc = 0.0, 0.0
            sample_count = 0

            self.optimizer.zero_grad()

            if self.config.BATCH_SIZE == 1:
                # Single sample training
                random.shuffle(dataset)
                for i, (state, label) in enumerate(dataset):
                    loss, acc, outputs, target = self._train_single_batch(state, label, i)

                    running_loss += loss
                    running_acc += acc
                    step += 1

                    if step % self.config.LOG_INTERVAL == 0:
                        avg_loss = running_loss / self.config.LOG_INTERVAL
                        avg_acc = running_acc / self.config.LOG_INTERVAL
                        self.monitor.update(
                            step, avg_loss, avg_acc,
                            state=state, target=target.item(),
                            pred=outputs.argmax(1).item()
                        )
                        running_loss, running_acc = 0.0, 0.0
            else:
                # Batch training
                for i, (state_batch, label_batch) in enumerate(dataloader):
                    loss, acc, outputs, targets = self._train_batch(state_batch, label_batch)

                    running_loss += loss * state_batch.size(0)
                    running_acc += acc
                    sample_count += state_batch.size(0)
                    step += 1

                    if step % self.config.LOG_INTERVAL == 0:
                        avg_loss = running_loss / sample_count
                        avg_acc = running_acc / sample_count
                        self.monitor.update(
                            step, avg_loss, avg_acc,
                            state=state_batch[0], target=targets[0].item(),
                            pred=outputs.argmax(1).tolist()[0]
                        )
                        running_loss, running_acc = 0.0, 0.0
                        sample_count = 0

            # Save checkpoint
            if epoch % self.config.SAVE_EVERY == 0 or epoch == self.config.EPOCHS:
                checkpoint_path = self._save_checkpoint(epoch)
                elapsed_time = time.time() - starttime
                remaining_epochs = self.config.EPOCHS - epoch - self.start_epoch

                print(f"Saved checkpoint: {checkpoint_path}")
                print(f"Time elapsed: {self.utils.format_seconds(int(elapsed_time))}")
                if remaining_epochs > 0:
                    predicted_time = (elapsed_time / epoch) * remaining_epochs
                    print(f"Runtime prediction: {self.utils.format_seconds(int(predicted_time))}")

        print("Training completed!")

def train():
    trainer = NeuralNetworkTrainer()
    trainer.train()

if __name__ == "__main__":
    train()