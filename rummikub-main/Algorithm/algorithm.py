import numpy as np
import random
import tensorflow as tf
import itertools
from tensorflow import keras
from keras import layers

class Tile:
    def __init__(self, number, color):
        self.number = number
        self.color = color

    def __repr__(self):
        return f"{self.color}{self.number}"

def create_tile_set():
    colors = ['Red', 'Blue', 'Yellow', 'Black']
    tiles = [Tile(number, color) for color in colors for number in range(1, 14)]
    tiles.extend(tiles)  # Duplicate the set
    tiles.append(Tile(0, 'Joker'))  # Adding two jokers
    tiles.append(Tile(0, 'Joker'))
    random.shuffle(tiles)
    return tiles

def distribute_tiles(tiles, num_players):
    hand_size = {2: 14, 3: 13, 4: 13}.get(num_players, 13)
    player_hands = {player: [tiles.pop() for _ in range(hand_size)] for player in range(1, num_players + 1)}
    return player_hands, tiles

class RummikubEnvironment:
    def __init__(self):
        self.num_players = 4  # Example number of players
        self.tiles = create_tile_set()
        self.player_hands, self.draw_pile = distribute_tiles(self.tiles, self.num_players)
        self.board_state = []  # Initialize board_state here
        self.current_player = 0 # Initialize current_player
        self.has_initial_move = [False] * self.num_players  # Initialize has_initial_move
        
    def reset(self):
        self.tiles = create_tile_set()
        self.player_hands, self.draw_pile = distribute_tiles(self.tiles, self.num_players)
        self.board_state = []
        self.current_player = 0
        self.has_initial_move = [False] * self.num_players  # Track if the initial move is made
        state = self.get_state_representation()
        return state

    def get_state_representation(self):
        # Player hands (2D to 1D)
        player_hands_representation = np.array([[0 for _ in range(106)] for _ in range(self.num_players)]).flatten()

        # Board state (1D)
        board_state_representation = np.array([0 for _ in range(106)])
        for tile in self.board_state:
            index = self.tile_to_index(tile)
            board_state_representation[index] += 1

        # Draw pile (single value)
        draw_pile_representation = np.array([len(self.draw_pile)])

        # Current player (1D)
        current_player_representation = np.array([1 if i == self.current_player else 0 for i in range(self.num_players)])

        # Initial move (1D)
        initial_move_representation = np.array([1 if has_moved else 0 for has_moved in self.has_initial_move])
        
        # Combine all parts of the state into a single 1D array
        state = np.concatenate([
            player_hands_representation, 
            board_state_representation, 
            draw_pile_representation, 
            current_player_representation, 
            initial_move_representation
        ])
        return state


    def tile_to_index(self, tile):
        # Convert a tile to an index in the representation
        color_index = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Black': 3, 'Joker': 4}.get(tile.color)
        number_index = tile.number
        if color_index < 4:  # For non-Joker tiles
            return color_index * 13 + (number_index - 1)
        else:  # For Joker tiles
            return 104 + color_index - 4

    def step(self, action):
        # Apply an action and update the game state
        reward = 0
        done = False
        info = {}
        
        # Check if the action is valid
        if not self.is_valid_action(action):
            # Penalize invalid actions
            reward = -1
            self.next_player()
        else:
            if not self.has_initial_move[self.current_player]:
                if self.is_valid_initial_move(action):
                    self.apply_action(action)
                    self.has_initial_move[self.current_player] = True
                    reward = 1  # Reward for successful initial move
                else:
                    # Player must draw a tile if the initial move is not valid
                    self.player_hands[self.current_player].append(self.draw_pile.pop())
                    reward = -1  # Penalty for failing to make initial move
            else:
                # Regular gameplay logic
                self.apply_action(action)
                # Define reward based on action success, points scored, etc.
                reward = self.calculate_reward(action)

            # Check if the game is finished
            done = self.is_game_done()
            self.next_player()

        # Update the game state
        next_state = self.get_state_representation()

        return next_state, reward, done, info
        
    def is_valid_run(self, tiles):
        # Check if the tiles form a valid run (consecutive numbers, same color)
        if len(set(tile.color for tile in tiles)) != 1:
            # All tiles must be of the same color
            return False

        # Sort tiles by number to check for consecutive numbers
        numbers = sorted(tile.number for tile in tiles)
        return all(numbers[i] + 1 == numbers[i + 1] for i in range(len(numbers) - 1))

    def is_valid_group(self, tiles):
        # Check if the tiles form a valid group (same number, different colors)
        if len(set(tile.number for tile in tiles)) != 1:
            # All tiles must be of the same number
            return False

        if len(tiles) > 4:
            # A group can't have more than 4 tiles
            return False

        # All colors must be different
        colors = [tile.color for tile in tiles]
        return len(colors) == len(set(colors))
    

    def is_valid_initial_move(self, action):
        return True
        
    def is_valid_action(self, action):
         # Assuming 'action' is a list of Tile objects the player wants to play
        if not action:
            # If the action is an empty list, it's not a valid play
            return False

        if len(action) < 3:
            # In Rummikub, a valid set must have at least 3 tiles
            return False

        # Check if the action is a valid run
        if self.is_valid_run(action):
            return True

        # Check if the action is a valid group
        if self.is_valid_group(action):
            return True

        # If neither, the action is not valid
        return False

    def apply_action(self, action):
        # 'action' could be a list of Tile objects from the player's hand
        # or a complex action involving both hand and board tiles
        if self.is_action_placing_new_set(action):
            # Place the new set from hand to board
            self.place_new_set_on_board(action)
        elif self.is_action_forming_combinations(action):
            # Form new combinations using board and hand tiles
            self.form_combinations_with_board(action)
        else:
            # The action is not valid
            raise ValueError("Invalid action")

    def is_action_placing_new_set(self, action):
        # Check if the action is placing a new set from hand
        if action.get("type") == "new_set" and "tiles_from_hand" in action:
            # Check if the tiles form a valid set
            return self.is_valid_action(action["tiles_from_hand"])
        return False

    def is_action_forming_combinations(self, action):
        # Check if the action involves forming combinations with board tiles
        # Assuming action is a dictionary
        if action.get("type") == "combination" and "board_manipulation" in action:
            # Additional checks can be added here to validate the structure of the manipulation
            return True
        return False

    def place_new_set_on_board(self, tiles):
        # Assume tiles is a list of Tile objects forming a valid set
        # Remove these tiles from the player's hand and add them to the board
        for tile in tiles:
            self.player_hands[self.current_player].remove(tile)
        self.board_state.extend(tiles)

    def form_combinations_with_board(self, action):
        # This method should handle complex manipulations involving
        # both the player's hand and existing sets on the board
        # Validate the resulting board state
        if not self.is_board_state_valid():
            # If the resulting board state is invalid, revert changes
            self.revert_board_and_apply_penalty()
            return

    def is_board_state_valid(self):
        # Check if all sets on the board are valid
        # This involves iterating over all sets on the board
        for set in self.board_state:
            if not (self.is_valid_run(set) or self.is_valid_group(set)):
                return False
        return True
    
    def next_player(self):
        # This method is called to switch to the next player
        # Before switching, store the current board state
        self.initial_board_state = self.board_state.copy()

    def revert_board_and_apply_penalty(self):
        # Revert the board to its state at the beginning of the turn
        self.board_state = self.initial_board_state.copy()

        # Apply penalty: Draw a tile from the draw pile
        if self.draw_pile:
            drawn_tile = self.draw_pile.pop()
            self.player_hands[self.current_player].append(drawn_tile)
        else:
            # Handle case where draw pile is empty, if needed
            pass

    def is_game_done(self):
        # Check if any player has no tiles left in their hand
        for player_hand in self.player_hands.values():
            if len(player_hand) == 0:
                return True  # Game ends if any player has no tiles left

        return False  # Game continues if all players have tiles

    def state_size(self):
        # The state size will be the total length of the state representation array
        return len(self.get_state_representation())

    def action_size(self):
        # Define the size of the action space
        # For simplicity, let's assume each player can either play a tile or draw from the pile
        # You'll likely need to expand this to account for the actual complexity of the actions in Rummikub
        return 2  # Placeholder, replace with your game's logic
        
    # Implement the following methods based on Rummikub rules
    def apply_penalty(self):
        # Logic for applying penalty to the player
        pass

    def apply_initial_meld(self, action):
        # Assuming 'action' is a list of Tile objects the player wants to play
        # Check if the total value of tiles is at least 30
        if sum(tile.number for tile in action) < 30:
            return False, "The total value of the tiles is less than 30."

        # Check if the action forms a valid group or run
        if not (self.is_valid_run(action) or self.is_valid_group(action)):
            return False, "The tiles do not form a valid group or run."

        # Place the tiles on the board
        self.place_new_set_on_board(action)
        
        return True, "Initial meld successful."

        
    def is_valid_meld_or_manipulation(self, player, action):
        # This method checks if a player's action is a valid initial meld
        # First, check if it's the player's first move
        if not self.has_initial_move[player]:
            # If it's the first move, ensure the action meets the initial meld requirement
            if not self.is_initial_meld_valid(action):
                return False
            # Mark that the player has made their initial move
            self.has_initial_move[player] = True
        else:
            # If not the first move, apply standard action validity checks (for subsequent turns) 
            if not self.is_valid_standard_action(player, action):
                return False
        return True
    
    def is_valid_standard_action(self, player, action):
        # Assuming 'action' is a representation of the player's move
        if self.is_action_placing_new_set(action):
            # Check if the new set formed from the hand is valid
            if not self.is_valid_set(action["tiles_from_hand"]):
                return False
        elif self.is_action_forming_combinations(action):
            # Apply the action to form combinations (this should be a hypothetical application,
            # i.e., not actually changing the game state yet)
            hypothetical_board = self.form_combinations_with_board(action, hypothetical=True)

            # Check if all sets on the hypothetical board are valid
            if not all(self.is_valid_set(set) for set in hypothetical_board):
                return False
        else:
            # If the action is neither placing a new set nor forming combinations, it's invalid
            return False
        return True

    def is_valid_set(self, tiles):
        # Check if a set of tiles is a valid group or run
        return self.is_valid_run(tiles) or self.is_valid_group(tiles)
    
    def is_initial_meld_valid(self, action):
        # Check if the action forms a valid run or group
        if not (self.is_valid_run(action) or self.is_valid_group(action)):
            return False

        # Check if the total value of the action is at least 30
        if sum(tile.number for tile in action) < 30:
            return False

        return True

    def apply_meld_or_manipulation(self, player, action):
        if action["type"] == "new_set":
            self.add_new_set(player, action["tiles_from_hand"])

        elif action["type"] == "manipulate":
            self.rearrange_board(player, action["manipulation_details"])

        elif action["type"] == "draw":
            self.draw_tile(player)


    def add_new_set(self, player, tiles):
        # Remove tiles from player's hand and add them to the board
        for tile in tiles:
            self.player_hands[player].remove(tile)
        self.board_state.append(tiles)  # Assuming board_state is a list of sets

    def rearrange_board(self, player, manipulation_details):
        # Update the board state based on the manipulation details
        # Process additions
        for set_index, tile in manipulation_details.get("add", []):
            self.add_tile_to_set(set_index, tile)

        # Process splits
        for set_index in manipulation_details.get("split", []):
            self.split_set(set_index)

        # Process merges
        for set_indices in manipulation_details.get("merge", []):
            self.merge_sets(*set_indices)

    def add_tile_to_set(self, set_index, tile, player):
        # Adds a tile to an existing set on the board
        self.board_state[set_index].append(tile)
        self.player_hands[player].remove(tile)

    def split_set(self, set_index, split_index):
        # Splits the set at 'set_index' into two sets at 'split_index'
        original_set = self.board_state[set_index]
        
        # Ensure the split index is valid
        if split_index <= 0 or split_index >= len(original_set):
            raise ValueError("Invalid split index")

        # Create two new sets from the split
        new_set1 = original_set[:split_index]
        new_set2 = original_set[split_index:]

        # Check if both new sets are valid
        if not (self.is_valid_set(new_set1) and self.is_valid_set(new_set2)):
            raise ValueError("Invalid split resulting in invalid sets")

        # Replace the original set with the new sets
        self.board_state[set_index] = new_set1
        self.board_state.insert(set_index + 1, new_set2)


    def merge_sets(self, set_index1, set_index2):
        # Merges two sets into one
        new_set = self.board_state[set_index1] + self.board_state[set_index2]
        self.board_state[set_index1] = new_set
        self.board_state.pop(set_index2)

    def draw_tile(self, player):
        # Player draws a tile from the draw pile if available
        if self.draw_pile:
            drawn_tile = self.draw_pile.pop()
            self.player_hands[player].append(drawn_tile) 

    def check_if_player_won(self):
        # A player wins if they have no more tiles in their hand
        return len(self.player_hands[self.current_player]) == 0

    def is_game_done(self):
        # The game ends if the pool is empty and the current player cannot play
        if not self.draw_pile and not self.can_player_play():
            return True
        return False

    def can_player_play(self, player):
        # Check if the player can form a valid set with their hand
        if self.can_form_valid_set_from_hand(player):
            return True

        # Check if the player can manipulate existing sets on the board in a valid way
        if self.can_manipulate_existing_sets(player):
            return True

        return False

    def can_form_valid_set_from_hand(self, player):
        # Check all combinations of tiles in the player's hand to form a valid set
        hand = self.player_hands[player]
        for num_tiles in range(3, len(hand) + 1):
            for subset in itertools.combinations(hand, num_tiles):
                if self.is_valid_run(list(subset)) or self.is_valid_group(list(subset)):
                    return True
        return False

    def can_manipulate_existing_sets(self, player):
        # Check if the player can add any of their tiles to existing sets on the board
        hand = self.player_hands[player]
        for tile in hand:
            for board_set in self.board_state:
                if self.can_tile_be_added_to_set(tile, board_set):
                    return True
        return False

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.epsilon = 1.0  # High initial exploration rate
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.epsilon_decay = 0.995  # Decay rate

    def _build_model(self):
        # Build a neural network model
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: Choose a random action
            action_index = random.randrange(self.action_size)
            action = self.convert_to_action_format(action_index)
        else:
            # Exploitation: Choose the best action based on Q-values
            flat_state = np.array(state).flatten()
            flat_state = np.reshape(flat_state, [1, self.state_size])
            q_values = self.model.predict(flat_state)
            action_index = np.argmax(q_values[0])
            action = self.convert_to_action_format(action_index)
        
        return action

    # Decay epsilon after each episode
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def convert_to_action_format(self, action_index):
        # Assuming the first N indices correspond to tile actions and the rest to draw actions
        TILE_ACTION_THRESHOLD = 100  # Example threshold

        if action_index < TILE_ACTION_THRESHOLD:
            return self.interpret_tile_action(action_index)
        else:
            return self.interpret_draw_action(action_index)

    def interpret_tile_action(self, action_index):
        tile_action_map = {
            0: [Tile(1, 'Red'), Tile(2, 'Red')],
            1: [Tile(3, 'Blue'), Tile(4, 'Blue')],
            # ... other mappings
        }
        return tile_action_map.get(action_index, [])

    def interpret_draw_action(self, action_index):
        # For drawing a tile, the action might simply be a 'draw' command
        return "draw"

    def learn(self, state, action, reward, next_state, done):
        # Convert state and next_state for neural network
        flat_state = np.array(state).flatten().reshape([1, self.state_size])
        flat_next_state = np.array(next_state).flatten().reshape([1, self.state_size])

        # Convert the complex action back to its index
        action_index = self.action_to_index(action)

        # Calculate target value
        target = reward
        if not done:
            target += 0.95 * np.amax(self.model.predict(flat_next_state)[0])  # discount factor

        # Update the target for the action taken
        target_f = self.model.predict(flat_state)
        target_f[0][action_index] = target

        # Train the model
        self.model.fit(flat_state, target_f, epochs=1, verbose=0)

    def action_to_index(self, action):
        # Convert the action (which might be a list) to a tuple for hashing
        action_tuple = tuple(action) if isinstance(action, list) else action

        # Mapping of actions to indices
        # Ensure keys are tuples or other immutable types
        action_index_map = {
            "draw": 0,
            (Tile(1, 'Red'), Tile(2, 'Red')): 1,
            (Tile(3, 'Blue'), Tile(4, 'Blue')): 2,
            # ... other mappings
        }

        # Find the action in the map and return its index
        return action_index_map.get(action_tuple, -1)  # Return -1 or some default value for unknown actions

# Initialize environment and agent
env = RummikubEnvironment()
state_size = env.state_size()
action_size = env.action_size()
agent = DQNAgent(state_size, action_size)

# Define the number of episodes
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500

# training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        steps += 1

        if steps % 100 == 0:  # Log every 100 steps
            print(f"Episode {episode}, Step {steps}: State={state}, Action={action}, Reward={reward}")

    agent.update_epsilon()
    if episode % 10 == 0:  # Log after every 10 episodes
        print(f"Completed Episode {episode}")