import numpy as np
import math
from collections import deque

# this class provides a a possible encoding / decoding mechanism for game board states.


class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        TODO: You need to implement this.
        """

        # unpack the tuple
        col, row = cr

        return (self.N_COLS * row) + col

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        # Use integer divion and modulo to get the row and column
        row = n // self.N_COLS
        col = n % self.N_COLS

        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """
        # Check if valid, then  return termination state
        if self.is_valid():
            return (self.state[5] >= 49 or self.state[11] <= 6)
        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        
        TODO: You need to implement this.
        """
        # Initial ball positions
        w_arr, b_arr = self.state[0:5], self.state[6:11]
        combined_arr = np.concatenate((w_arr, b_arr))
        white_ball_pos = self.state[5]
        black_ball_pos = self.state[11]
        balls = [white_ball_pos, black_ball_pos]
       # Check if every element in combined arr is unique
        if len(combined_arr) != len(set(combined_arr)):
            return False
        # Check if every element in balls arr is unique
        if len(balls) != len(set(balls)):
            return False
        # Check valid max and min positions in the board
        max_pos = max(self.state)
        min_pos = min(self.state)
        if max_pos > 55 or min_pos < 0:
            return False
        # White ball check
        max_pos = max(w_arr)
        min_pos = min(w_arr)
        if white_ball_pos < min_pos:
            return False
        if white_ball_pos > max_pos:
            return False
        # Black ball check
        min_pos = min(b_arr)
        max_pos = max(b_arr)
        if black_ball_pos < min_pos:
            return False
        if black_ball_pos > max_pos:
            return False
        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.

        TODO: You need to implement this.
        """
        total_possible_actions = []
        # Decode the current position
        x_pos, y_pos = board_state.decode_single_pos(
            board_state.state[piece_idx])
        w_arr, b_arr = board_state.state[0:5], board_state.state[6:11]
        white_ball_pos = board_state.state[5]
        black_ball_pos = board_state.state[11]
        # Check if the piece is a ball
        if (board_state.state[piece_idx] == white_ball_pos) or (board_state.state[piece_idx] == black_ball_pos):
            return total_possible_actions
        # Get the corresponding opposing array
        opp_arr = []
        our_arr = []
        if piece_idx < 6:
            opp_arr = b_arr
            our_arr = w_arr
        else:
            opp_arr = w_arr
            our_arr = b_arr
        # Up check
        new_y_pos = y_pos + 2
        new_x_pos_left = x_pos - 1
        new_x_pos_right = x_pos + 1
        if new_y_pos < board_state.N_ROWS:
            if new_x_pos_left >= 0:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos_left, new_y_pos))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(
                        board_state.encode_single_pos((new_x_pos_left, new_y_pos)))
            if new_x_pos_right < board_state.N_COLS:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos_right, new_y_pos))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(encoded_pos)
        # Down check
        new_y_pos = y_pos - 2
        new_x_pos_left = x_pos - 1
        new_x_pos_right = x_pos + 1
        if new_y_pos >= 0:
            if new_x_pos_left >= 0:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos_left, new_y_pos))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(
                        board_state.encode_single_pos((new_x_pos_left, new_y_pos)))
            if new_x_pos_right < board_state.N_COLS:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos_right, new_y_pos))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(encoded_pos)
        # Right check
        new_x_pos = x_pos + 2
        new_y_pos_up = y_pos + 1
        new_y_pos_down = y_pos - 1
        if new_x_pos < 7:
            if new_y_pos_down >= 0:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos, new_y_pos_down))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(
                        board_state.encode_single_pos((new_x_pos, new_y_pos_down)))
            if new_y_pos_up < board_state.N_ROWS:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos, new_y_pos_up))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(encoded_pos)
        # Left check
        new_x_pos = x_pos - 2
        new_y_pos_up = y_pos + 1
        new_y_pos_down = y_pos - 1
        if new_x_pos >= 0:
            if new_y_pos_down >= 0:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos, new_y_pos_down))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(
                        board_state.encode_single_pos((new_x_pos, new_y_pos_down)))
            if new_y_pos_up < board_state.N_ROWS:
                encoded_pos = board_state.encode_single_pos(
                    (new_x_pos, new_y_pos_up))
                if encoded_pos not in opp_arr and encoded_pos not in our_arr:
                    total_possible_actions.append(encoded_pos)
        return total_possible_actions

    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx in the board_state.

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.

        TODO: You need to implement this.
        """
        # Get the right arrays
        ball_to_use = 0
        w_arr, b_arr = board_state.state[0:5], board_state.state[6:11]
        enemy_arr = []
        our_arr = []
        if player_idx == 0:
            # White player moves
            ball_to_use = board_state.state[5]
            enemy_arr = b_arr
            our_arr = w_arr
            if ball_to_use not in our_arr:
                return set()
            return Rules.find_neighbors(board_state, ball_to_use, our_arr, enemy_arr)
        else:
            # Black player moves
            ball_to_use = board_state.state[11]
            enemy_arr = w_arr
            our_arr = b_arr
            if ball_to_use not in our_arr:
                return set()
            return Rules.find_neighbors(board_state, ball_to_use, our_arr, enemy_arr)

    def find_neighbors(board_state, original_ball_pos, our_arr, enemy_arr):
        # Help er method to find the neighbors
        queue = []
        visited = []
        queue.append(original_ball_pos)

        while queue:
            ball = queue.pop(0)
            neighbors = Rules.check_directions(
                board_state, our_arr, enemy_arr, ball, visited)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.append(neighbor)
        # remove the starting ball position from the visited array
        if original_ball_pos in visited:
            visited.remove(original_ball_pos)
        return set(visited)

    def check_directions(board_state, our_arr, enemy_arr, cur_ball_pos, visited):
        # Helper method to check the directions
        all_possible_neighbors = []
        x, y = board_state.decode_single_pos(cur_ball_pos)

        directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                      (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # Iterate over the potential directions
        for direction in directions:
            cur_x = x
            cur_y = y
            while (cur_x >= 0 and cur_x < board_state.N_COLS and cur_y >= 0 and cur_y < board_state.N_ROWS):
                cur_x += direction[0]
                cur_y += direction[1]
                if cur_x >= 0 and cur_x < board_state.N_COLS and cur_y >= 0 and cur_y < board_state.N_ROWS:
                    encoded_pos = board_state.encode_single_pos((cur_x, cur_y))
                    if encoded_pos in enemy_arr:
                        break
                    elif encoded_pos in visited:
                        break
                    elif encoded_pos in our_arr:
                        all_possible_neighbors.append(encoded_pos)
                        break
                    else:
                        continue
        return all_possible_neighbors


class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.

        TODO: You need to implement this.
        """
        if player_idx == 0:
            # White piece plays
            white_pieces = self.game_state.state[0:5]
            generated_actions = GameSimulator.generate_actions_for_piece(
                self, player_idx, white_pieces)
            return generated_actions
        else:
            # Black piece plays
            black_pieces = self.game_state.state[6:11]
            generated_actions = GameSimulator.generate_actions_for_piece(
                self, player_idx, black_pieces)
            return generated_actions
    
    def generate_actions_for_piece(self, player_idx: int, pieces):
        valid_piece_moves = []
        offset = player_idx * 6
        for idx, _ in enumerate(pieces):
            valid_moves_for_curr_piece = Rules.single_piece_actions(self.game_state, idx + offset)
            for move in valid_moves_for_curr_piece:
                valid_piece_moves.append((idx, move))
        valid_ball_moves = Rules.single_ball_actions(self.game_state, player_idx)
        for move in valid_ball_moves:
            valid_piece_moves.append((5, move))
        return valid_piece_moves

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        TODO: You need to implement this.
        """
        relative_idx, single_action = action
        if relative_idx == 5:
            valid_ball_moves = Rules.single_ball_actions(
                self.game_state, player_idx)
            if single_action in valid_ball_moves:
                return True
            else:
                raise ValueError("Ball move is not valid.")
        else:
            valid_piece_moves = Rules.single_piece_actions(
                self.game_state, relative_idx + player_idx * 6)
            if single_action in valid_piece_moves:
                return True
            else:
                raise ValueError("Piece move is not valid.")
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)

"""
All classes below help us to create different types of players that play the game in test_search.
"""

# this serves as a base class from which we will subclass to create different 
# types of players that play the game.
class Player:
    def __init__(self, policy_fnc):
        self.policy_fnc = policy_fnc
    def policy(self, decode_state): 
        pass

class RandomPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.random_algorithm)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        state_tup = tuple((encoded_state_tup, self.player_idx))
        return self.policy_fnc(state_tup)

class PassivePlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.passive_algorithm)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        state_tup = tuple((encoded_state_tup, self.player_idx))
        return self.policy_fnc(state_tup)

class MiniMaxPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.mini_max_search)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        self.b.state = encoded_state_tup
        self.b.decode_state = self.b.make_state()
        alpha = -math.inf
        beta = math.inf
        depth = 3
        return self.policy_fnc(self.b, self.player_idx, alpha, beta, depth)
    
class NegaMaxPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.nega_max_search)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        self.b.state = encoded_state_tup
        self.b.decode_state = self.b.make_state()
        alpha = -math.inf
        beta = math.inf
        depth = 3
        return self.policy_fnc(self.b, self.player_idx, alpha, beta, depth)

class IterativeDeepPVSPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.principle_variation_iterative_deepening)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        self.b.state = encoded_state_tup
        self.b.decode_state = self.b.make_state()
        return self.policy_fnc(self.b, self.player_idx, 4)

# iterative deep player uses Principle Variation Algorithm
# Principle Variation Algorithm can be used to find the best move alone, does well @ 4.
class PrincipleVariationPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.principle_variation)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state) 
        self.b.state = encoded_state_tup
        self.b.decode_state = self.b.make_state()
        alpha = -math.inf
        beta = math.inf
        depth = 3
        return self.policy_fnc(self.b, self.player_idx, depth, alpha, beta)