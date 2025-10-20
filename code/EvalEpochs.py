import ast
import NNEvaluation
import NeuralNetData
import torch
import rules
import matplotlib.pyplot as plt

TEST_PATH = r"D:\Matura\NNnewer\sepember\minimax_training_set_symmetries.txt"
NN_PATH = r'checkpoints_batch512'
GAME_AMOUNT = 1000
MAX_EPOCH = 2090
EPOCH_STEP = 100

def eval_epoch_test(epoch):
    policy_net = NeuralNetData.BiggerPolicyNetwork(state_size=28, action_size=88)
    policy_net.load_state_dict(torch.load(f'{NN_PATH}/model_epoch{epoch}.pth'))
    policy_net.eval()
    actions = rules.get_actions()

    correct = 0
    total = 0

    with open(TEST_PATH, "r") as f:
        start_line = 500000
        end_line = 527000
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= end_line:
                break
            state, move = ast.literal_eval(line.strip())

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                output = policy_net(state_tensor)
                _, predicted = torch.max(output.data, 1)
                total += 1
                if actions[predicted.item()] == move:
                    correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f'Epoch {epoch}: Accuracy on test set: {accuracy * 100:.2f}% ({correct}/{total})')

    return accuracy

def eval_epoch_play(epoch, opponent):
    policy_net = NeuralNetData.BiggerPolicyNetwork(state_size=28, action_size=88)
    policy_net.load_state_dict(torch.load(f'{NN_PATH}/model_epoch{epoch}.pth'))
    policy_net.eval()

    if opponent == 'random':
        player1wins, player2wins, draws = NNEvaluation.match('2', '4', GAME_AMOUNT, policy_net)[2:]
    if opponent == 'minimax':
        player1wins, player2wins, draws = NNEvaluation.match('2', '1', GAME_AMOUNT, policy_net, opening_random_moves=10)[2:]
    print(f'Epoch {epoch}: NN Wins: {player1wins}, {opponent} Wins: {player2wins}, Draws: {draws}')

    return player1wins, player2wins, draws


def graph_results_test(accuracies, epochs, title_suffix=''):
    plt.figure(figsize=(12, 6))  # Make figure wider to accommodate labels
    plt.plot(epochs, accuracies, marker='o', linewidth=2, markersize=4)
    plt.title(f'NN Test Set Accuracy Over Epochs{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Set x-axis ticks to show every few epochs to avoid crowding
    if len(epochs) > 10:
        tick_step = max(1, len(epochs) // 10)  # Show about 10 ticks
        plt.xticks(epochs[::tick_step], rotation=45)
    else:
        plt.xticks(epochs)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Ensure labels don't get cut off
    plt.savefig('nn_test_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

def graph_results_play(player1wins, player2wins, draws, epochs, title_suffix: str = ''):
    # Choose distinct colors
    colors = {
        "NN Wins": "tab:blue",
        "Opponent Wins": "tab:red",
        "Draws": "tab:green"
    }

    # Make sure inputs are valid
    if not (len(player1wins) == len(player2wins) == len(draws) == len(epochs)):
        raise ValueError("All input lists must be of the same length.")

    # Calculate appropriate bar width based on epoch spacing
    if len(epochs) > 1:
        epoch_spacing = epochs[1] - epochs[0]
        bar_width = epoch_spacing * 0.8  # Use 80% of the spacing for bar width
    else:
        bar_width = 80  # Default width if only one epoch

    # Create the bar chart with larger figure size
    plt.figure(figsize=(14, 8))

    # Plot stacked bars with explicit width
    plt.bar(epochs, player1wins, width=bar_width, color=colors["NN Wins"], 
            label="NN Wins", edgecolor='black', linewidth=0.5)
    plt.bar(epochs, player2wins, width=bar_width, bottom=player1wins, 
        color=colors["Opponent Wins"], label="Opponent Wins", 
            edgecolor='black', linewidth=0.5)
    plt.bar(epochs, draws, width=bar_width, 
            bottom=[p1 + p2 for p1, p2 in zip(player1wins, player2wins)],
            color=colors["Draws"], label="Draws", 
            edgecolor='black', linewidth=0.5)

    # Labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Game Results")
    plt.title(f"Training Results Over Epochs{title_suffix}")
    plt.legend()

    # Set x-axis ticks to show every few epochs to avoid crowding
    if len(epochs) > 10:
        tick_step = max(1, len(epochs) // 10)  # Show about 10 ticks
        plt.xticks(epochs[::tick_step], rotation=45)
    else:
        plt.xticks(epochs)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    accuracies = []
    p1w_r = []
    p2w_r = []
    d_r = []
    p1w_m = []
    p2w_m = []
    d_m = []
    epochs = list(range(0, MAX_EPOCH + 1, EPOCH_STEP))

    for epoch in epochs:
        if epoch == 0:
            accuracies.append(0)
            p1w_r.append(0)
            p2w_r.append(0)
            d_r.append(0)
            p1w_m.append(0)
            p2w_m.append(0)
            d_m.append(0)
            continue
        accuracy = eval_epoch_test(epoch)
        accuracies.append(accuracy)

        p1w, p2w, d = eval_epoch_play(epoch, 'random')
        p1w_r.append(p1w)
        p2w_r.append(p2w)
        d_r.append(d)
        p1w, p2w, d = eval_epoch_play(epoch, 'minimax')
        p1w_m.append(p1w)
        p2w_m.append(p2w)
        d_m.append(d)

    graph_results_test(accuracies, epochs)
    graph_results_play(p1w_r, p2w_r, d_r, epochs, title_suffix=' vs Random')
    graph_results_play(p1w_m, p2w_m, d_m, epochs, title_suffix=' vs Minimax')