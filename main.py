import numpy as np
import pickle


class CurralGame:
    def __init__(self):
        self.board = self.initialize_board()
        self.game_over = False
        self.current_player = 1  # Player 1 starts

    def initialize_board(self):
        # Create a 5x5 board where Player 1 is '1' and Player 2 is '2'
        board = np.zeros((5, 5), dtype=int)
        # Example initial positions
        board[0, 0] = 1  # Player 1's piece
        board[0, 1] = 1  # Player 1's piece
        board[4, 3] = 2  # Player 2's piece
        board[4, 4] = 2  # Player 2's piece
        return board

    def reset(self):
        self.board = self.initialize_board()
        self.game_over = False
        self.current_player = 1
        return self.flatten_board(self.board)

    def flatten_board(self, board):
        return board.flatten()  # Convert the board to a 1D array

    def valid_moves(self):
        return valid_moves(self.board, self.current_player)

    def make_move(self, move):
        x, y, new_x, new_y = move
        self.board[new_x, new_y] = self.board[x, y]
        self.board[x, y] = 0
        return self.flatten_board(self.board)

    def is_game_over(self):
        """
        Check if the game is over:
        - The game is over if a player cannot make any valid moves.
        - The game is also over if a player has no pieces left or the opponent has no pieces left.
        """
        # Check if the current player has any pieces on the board
        player_pieces = np.count_nonzero(self.board == self.current_player)
        opponent_pieces = np.count_nonzero(self.board == (3 - self.current_player))  # Opponent is 1 if current player is 2, and vice versa

        # Game is over if the current player has no pieces left
        if player_pieces == 0:
            self.game_over = True
            return True

        # Game is over if the opponent has no pieces left (current player wins)
        if opponent_pieces == 0:
            self.game_over = True
            return True

        # Check if the current player has any valid moves
        if len(self.valid_moves()) == 0:
            self.game_over = True
            return True

        # Game is not over
        self.game_over = False
        return False

    def get_reward(self):
        """Calculate the reward based on the game mechanics."""
        reward = 0
        # Reward or penalty based on game state
        opponent = 2 if self.current_player == 1 else 1
        
        # 1. Check for piece captures
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] == self.current_player:
                    # Check if the current player captured an opponent's piece
                    if (x > 0 and self.board[x-1, y] == opponent) or \
                       (x < self.board.shape[0] - 1 and self.board[x+1, y] == opponent) or \
                       (y > 0 and self.board[x, y-1] == opponent) or \
                       (y < self.board.shape[1] - 1 and self.board[x, y+1] == opponent):
                        reward += 1  # Reward for capturing a piece

        # 2. Penalize if the agent loses a piece
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] == opponent:
                    # Check if the opponent captured the current player's piece
                    if (x > 0 and self.board[x-1, y] == self.current_player) or \
                       (x < self.board.shape[0] - 1 and self.board[x+1, y] == self.current_player) or \
                       (y > 0 and self.board[x, y-1] == self.current_player) or \
                       (y < self.board.shape[1] - 1 and self.board[x, y+1] == self.current_player):
                        reward -= 1  # Penalty for losing a piece

        # 3. Reward for winning and penalize for losing
        if self.is_game_over():
            if self.current_player == 1:  # Assume player 1 wins (can refine this logic)
                reward += 10  # Large reward for winning
            else:
                reward -= 10  # Large penalty for losing

        return reward

def valid_moves(board, player):
    moves = []
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x, y] == player:
                # Check all possible directions and add valid moves
                if x > 0 and board[x-1, y] == 0:  # Up
                    moves.append((x, y, x-1, y))
                if x < board.shape[0] - 1 and board[x+1, y] == 0:  # Down
                    moves.append((x, y, x+1, y))
                if y > 0 and board[x, y-1] == 0:  # Left
                    moves.append((x, y, x, y-1))
                if y < board.shape[1] - 1 and board[x, y+1] == 0:  # Right
                    moves.append((x, y, x, y+1))
    return moves


class RLAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9):
        self.q_table = {}
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.action_space))

        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] = (self.q_table[state][action] +
                                       self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]))


def train(curral_game, agent, episodes=1000):
    for episode in range(episodes):
        print(f"Episode {episode+1} in {episodes}")
        state = curral_game.reset()
        total_reward = 0
        while not curral_game.is_game_over():
            action = agent.choose_action(str(state))  # Use string state for simplicity
            valid_actions = curral_game.valid_moves()
            move = valid_actions[action]
            next_state = curral_game.make_move(move)
            reward = curral_game.get_reward()
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Save the trained model
    with open("curral_agent_model.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)


def play_game(curral_game, agent):
    state = curral_game.reset()
    while not curral_game.is_game_over():
        # Human player makes a move
        print("Current board:")
        print(curral_game.board)
        player_move = input("Your move (format: x y to x' y'): ")
        # Parse the move and apply it to the board
        # Here you would implement parsing and applying the player's move to the board
        # For example:
        # player_move = (x, y, new_x, new_y)
        # curral_game.make_move(player_move)
        
        # Now it's the computer's (RL agent's) turn
        action = agent.choose_action(str(state))
        valid_actions = curral_game.valid_moves()
        move = valid_actions[action]
        state = curral_game.make_move(move)
        print(f"Computer moves: {move}")
    
    print("Game over")


# Initialize the game and train the agent
curral_game = CurralGame()
agent = RLAgent(curral_game.valid_moves(), learning_rate=0.1)

# Train the agent and save the model
train(curral_game, agent)

# After training, load the trained model
with open("curral_agent_model.pkl", "rb") as f:
    q_table = pickle.load(f)

# Initialize the agent with the loaded model
agent = RLAgent(curral_game.valid_moves(), learning_rate=0.1)
agent.q_table = q_table  # Load the trained Q-table

# Play the game (player vs computer)
play_game(curral_game, agent)
