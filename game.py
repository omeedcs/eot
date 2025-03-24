{{ ... }}
class GameSimulator:
    """
    Responsible for handling the game simulation between players.
    
    This class manages game state, validates moves, and updates the board
    as players take turns making moves.
    """

    def __init__(self, players):
        """
        Initialize a new game simulator.
        
        Args:
            players: List of player objects that will compete
        """
        self.game_state = BoardState()
        self.current_round = -1  # The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation until completion.
        
        Returns:
            Tuple containing (final_round, winner, message)
        """
        self.current_round = 0
        
        while True:
            player_idx = self.current_round % 2  # even - player 0 (white), odd - player 1 (black)
            
            ## Check if the game is in a termination state
            if self.game_state.is_termination_state():
                if self.game_state.state[5] >= 49:
                    return self.current_round, "WHITE", "White won the game!"
                else:
                    return self.current_round, "BLACK", "Black won the game!"
            
            ## Get action from the current player
            if self.players is not None:
                action = self.players[player_idx].policy(self.game_state.decode_state)
            else:
                return self.current_round, "NONE", "No players specified"
            
            ## Validate the action is legitimate
            try:
                self.validate_action(action, player_idx)
            except ValueError as e:
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black provided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)
            
            ## Increment the round number
            self.current_round += 1

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state and a player's turn, generate the set of possible actions.
        
        Args:
            player_idx: Which player (0 for white, 1 for black) is moving this turn
            
        Returns:
            Set of tuples (relative_idx, encoded position) representing all possible actions
        """
        if player_idx != 0 and player_idx != 1:
            raise ValueError("Player index must be 0 or 1")
            
        # List out all pieces, and filter by player_idx
        pieces = []
        # Player's normal pieces (blocks)
        for idx in range(5):
            pieces.append(idx)
        # Player's ball piece
        pieces.append(5)
        
        return self.generate_actions_for_piece(player_idx, pieces)
        
    def generate_actions_for_piece(self, player_idx: int, pieces):
        """
        Generate valid actions for specific pieces.
        
        Args:
            player_idx: Which player is moving
            pieces: List of relative indices of pieces to evaluate
            
        Returns:
            Set of valid actions for the specified pieces
        """
        if player_idx != 0 and player_idx != 1:
            raise ValueError("Player index must be 0 or 1")
        
        actions = set()
        offset = player_idx * 6
        
        # Player's normal pieces (blocks), add actions for each
        for rel_idx in pieces[:-1]:
            abs_idx = rel_idx + offset
            valid_pos = Rules.single_piece_actions(self.game_state, abs_idx)
            for pos in valid_pos:
                actions.add((rel_idx, pos))
                
        # Player's ball piece
        rel_ball_idx = pieces[-1]
        valid_ball_pos = Rules.single_ball_actions(self.game_state, player_idx)
        for pos in valid_ball_pos:
            actions.add((rel_ball_idx, pos))
            
        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether the specified action can be taken from this state.
        
        Args:
            action: Tuple (relative_idx, encoded position)
            player_idx: Integer 0 or 1 representing the player's turn
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate the action format
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple of (relative_idx, encoded position)")
            
        rel_idx, enc_pos = action
        
        # Validate the relative index
        if not isinstance(rel_idx, int) or rel_idx < 0 or rel_idx > 5:
            raise ValueError("Relative index must be an integer between 0 and 5 inclusive")
            
        # Validate the encoded position
        if not isinstance(enc_pos, int) or enc_pos < 0 or enc_pos > 55:
            raise ValueError("Encoded position must be an integer between 0 and 55 inclusive")
            
        # Check if it's in the set of valid actions
        act_set = self.generate_valid_actions(player_idx)
        
        if action not in act_set:
            raise ValueError("Action is not in the set of valid actions")
            
        return True

    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state.
        
        Args:
            action: Tuple (relative_idx, encoded position)
            player_idx: Which player is making the move (0 or 1)
        """
        rel_idx, enc_pos = action
        abs_idx = rel_idx + player_idx * 6
        self.game_state.update(abs_idx, enc_pos)


"""
Player classes help create different types of AI players for the game.
Each implements a different algorithm for selecting moves.
"""

class Player:
    """Base class for all players."""
    
    def __init__(self, policy_fnc):
        """
        Initialize a player with a policy function.
        
        Args:
            policy_fnc: Function that determines the player's next move
        """
        self.policy_fnc = policy_fnc
        
    def policy(self, decode_state):
        """
        Returns the player's chosen action based on the current state.
        
        Args:
            decode_state: The current decoded board state
            
        Returns:
            Chosen action as (relative_idx, encoded position)
        """
        return self.policy_fnc(decode_state)


class RandomPlayer(Player):
    """Player that makes random valid moves."""
    
    def __init__(self, gsp, player_idx):
        """
        Initialize a random player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.random_algorithm)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """Choose a random valid action."""
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc((self.b.state, self.player_idx))


class PassivePlayer(Player):
    """
    Player that only moves its ball, not its blocking pieces.
    """
    
    def __init__(self, gsp, player_idx):
        """
        Initialize a passive player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.passive_algorithm)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """Choose a random valid ball move."""
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc((self.b.state, self.player_idx))


class MiniMaxPlayer(Player):
    """
    Player that uses minimax algorithm with alpha-beta pruning.
    """
    
    def __init__(self, gsp, player_idx):
        """
        Initialize a minimax player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.mini_max_search)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """
        Choose the best action using minimax with alpha-beta pruning.
        
        Returns:
            Best action tuple and its score
        """
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc(self.b, self.player_idx, -math.inf, math.inf, 3)


class NegaMaxPlayer(Player):
    """
    Player that uses negamax algorithm with alpha-beta pruning.
    """
    
    def __init__(self, gsp, player_idx):
        """
        Initialize a negamax player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.nega_max_search)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """
        Choose the best action using negamax with alpha-beta pruning.
        
        Returns:
            Best action tuple and its score
        """
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc(self.b, self.player_idx, -math.inf, math.inf, 3)


class IterativeDeepPVSPlayer(Player):
    """
    Player that uses Iterative Deepening with Principal Variation Search.
    """
    
    def __init__(self, gsp, player_idx):
        """
        Initialize an iterative deepening PVS player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.principle_variation_iterative_deepening)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """
        Choose the best action using iterative deepening with PVS.
        
        Returns:
            Best action tuple and its score
        """
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc(self.b, self.player_idx, 4)


# PVS Player uses Principle Variation Algorithm without iterative deepening
class PrincipleVariationPlayer(Player):
    """
    Player that uses Principal Variation Search algorithm.
    """
    
    def __init__(self, gsp, player_idx):
        """
        Initialize a principal variation search player.
        
        Args:
            gsp: GameStateProblem instance
            player_idx: Player's index (0 or 1)
        """
        super().__init__(gsp.principle_variation)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        
    def policy(self, decode_state):
        """
        Choose the best action using principal variation search.
        
        Returns:
            Best action tuple and its score
        """
        self.b.decode_state = decode_state
        self.b.state = np.array([self.b.encode_single_pos(pos) for pos in decode_state])
        return self.policy_fnc(self.b, self.player_idx, 4, -math.inf, math.inf)