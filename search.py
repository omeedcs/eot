import numpy as np
import queue
from game import BoardState, GameSimulator, Rules
from queue import PriorityQueue
from math import dist
import math
import random

class Problem:
    """
    Abstract base class for defining search problems.
    
    Provides the interface for problem definitions used in search algorithms.
    """

    def __init__(self, initial_state, goal_state_set: set):
        """
        Initialize a search problem.
        
        Args:
            initial_state: The starting state
            goal_state_set: Set of valid goal states
        """
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state.
        
        Must be implemented by subclasses.
        
        Args:
            state: Current state to evaluate
            
        Returns:
            Iterable of valid actions
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action.
        
        Must be implemented by subclasses.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Resulting state after taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states.
        
        Args:
            state: State to check
            
        Returns:
            True if state is a goal state, False otherwise
        """
        return state in self.goal_state_set


class GameStateProblem(Problem):
    """
    Problem formulation for the game state search.
    
    Implements the interface defined by Problem to allow various search
    algorithms to find solutions to the game.
    """
    
    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        Initialize a game state problem.
        
        Args:
            initial_board_state: Starting board configuration
            goal_board_state: Target board configuration
            player_idx: Which player (0 or 1) is first to move
        """
        super().__init__(
            tuple((tuple(initial_board_state.state), player_idx)), 
            set([tuple((tuple(goal_board_state.state), 0)), 
                 tuple((tuple(goal_board_state.state), 1))])
        )
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        Set the search algorithm to use.
        
        Args:
            alg: Algorithm name (defaults to A* if unspecified)
        """
        self.search_alg_fnc = self.a_star_algorithm

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set of possible actions.
        
        Args: 
            state: Tuple (encoded_state, player_idx), where encoded_state is a 
                  tuple of 12 integers, and player_idx is the player moving this turn
                  
        Returns:
            Set of valid actions from this state
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action.
        
        Args: 
            state: Tuple (encoded_state, player_idx), where encoded_state is a 
                  tuple of 12 integers, and player_idx is the player moving this turn
            action: Tuple (relative_idx, position) describing the move
            
        Returns:
            Next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple(s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    def a_star_algorithm(self):
        """
        A* search algorithm implementation.
        
        Returns a list of (state, action) pairs from initial state to goal state.
        The first state and action correspond to the initial state and action,
        and the last pair has a goal state and None as the action.
        
        Returns:
            List of (state, action) pairs representing the solution path
        """
        # Assign the start and goal states
        start_state = self.initial_state
        goal_state = self.goal_state_set
        
        # Create the frontier, parent, and cost data structures
        frontier = PriorityQueue()
        frontier.put((0, start_state))
        parent = {}
        cost = {}
        
        # Assign the start state to the parent and cost
        parent[start_state] = None
        cost[start_state] = 0
        
        # Run A* search
        current_state = None
        while not frontier.empty():
            current_state = frontier.get()[1]
            if self.is_goal(current_state):
                break
                
            # Grab all actions from the current state
            actions = self.get_actions(current_state)
            for action in actions:
                # Execute the action
                next_state = self.execute(current_state, action)
                new_cost = cost[current_state] + 1
                
                # Check if the next state is the new best state
                if next_state not in cost or new_cost < cost[next_state]:
                    heuristic = new_cost
                    
                    # Find the euclidean distance between the current state and the goal state
                    for idx, piece in enumerate(next_state[0]):
                        goal_state_to_list = list(goal_state)
                        decode_func = BoardState().decode_single_pos
                        x, y = decode_func(goal_state_to_list[0][0][idx])
                        first_coord = [x, y]
                        x2, y2 = decode_func(piece)
                        second_coord = [x2, y2]
                        euc_dist = dist(first_coord, second_coord)
                        heuristic += euc_dist
                        
                    # Heuristic is the new cost + the euclidean distance computed above
                    frontier.put((heuristic, next_state))
                    cost[next_state] = new_cost
                    parent[next_state] = current_state
                    
        # Reconstruct the path
        goal_state_to_use = None
        goal_set_list = list(goal_state)
        goal_state_to_use = goal_set_list[goal_set_list.index(current_state)]
        node = goal_state_to_use
        
        # Iterate through the parents dictionary to find the right path
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
            
        # Reverse the path
        path.reverse()
        
        # Reconstruct the actions using the states
        result = []
        previous_state = path.pop(0)
        for p in path:
            # Find the move that was made between a path
            values, idx = self.find_mismatch(p[0], previous_state[0])
            idx = idx % 6
            # Create new state, action pairs
            state = previous_state
            action = (idx, values[0])
            result.append((state, action))
            previous_state = p

        # Append the final goal state to the result
        result.append((goal_state_to_use, None))
        return result

    def find_mismatch(self, next_state, prev_state):
        """
        Find the differences between two states.
        
        Given two arrays, finds the first mismatched value and returns 
        the value and index of the mismatch.
        
        Args: 
            next_state: Next state array
            prev_state: Previous state array
            
        Returns:
            Tuple of (list of mismatched values, index of mismatch)
        """
        values = []
        idx = -1
        for i, (x, y) in enumerate(zip(next_state, prev_state)):
            if x != y:
                values = [x, y]
                idx = i
        return values, idx

    def random_algorithm(self, enc_state):
        """
        Random player algorithm that takes random actions.
        
        Args:
            enc_state: Current encoded state
            
        Returns:
            Tuple of (chosen action, score=1)
        """
        actions = self.get_actions(enc_state)
        curr_action = random.choice(list(actions))
        return (curr_action, 1)
    
    def passive_algorithm(self, enc_state):
        """
        Passive player algorithm that only moves its ball.
        
        Args:
            enc_state: Current encoded state
            
        Returns:
            Tuple of (chosen ball action, score=1)
        """
        actions = self.get_actions(enc_state)
        ball_actions = []
        for action in actions:
            if action[0] == 5:
                ball_actions.append(action)
        curr_action = random.choice(ball_actions)
        return (curr_action, 1)

    def mini_max_search(self, board, player_idx, alpha, beta, depth):
        """
        Minimax search with alpha-beta pruning.
        
        Finds the best move for the player using the minimax algorithm
        with alpha-beta pruning.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Search depth
            
        Returns:
            Tuple of (best action, score)
        """
        best_action = None
        max_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        
        # Iterate through possible actions
        for action in actions:
            # Execute an action, create a board copy
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            # Finding the optimal score for the next level
            score = self.max_value(board_copy, player_idx, alpha, beta, depth)
            if score > max_score:
                # Assign best action accordingly
                best_action = action
                max_score = score
                
        return best_action, max_score

    def max_value(self, board, player_idx, alpha, beta, depth):
        """
        Helper function for minimax - max value calculation.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Remaining search depth
            
        Returns:
            Maximum score for this node
        """
        if depth == 0 or board.is_termination_state():
            # Maximizing player, no need to negate evaluation
            return self.evaluate_board(board, player_idx)
            
        max_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        
        for action in actions:
            # Execute an action, create a board copy
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            # Update the player index
            new_player_idx = 1 if player_idx == 0 else 0
            
            # Find the maximum value using the minimum recursion call
            max_score = max(max_score, self.min_value(board_copy, new_player_idx, alpha, beta, depth - 1))
            
            # Early pruning
            if max_score >= beta:
                return max_score
            alpha = max(alpha, max_score)
            
        return max_score
    
    def min_value(self, board, player_idx, alpha, beta, depth):
        """
        Helper function for minimax - min value calculation.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Remaining search depth
            
        Returns:
            Minimum score for this node
        """
        if depth == 0 or board.is_termination_state():
            # Minimizing player, negate the evaluation value
            return -self.evaluate_board(board, player_idx)
            
        min_score = math.inf
        actions = self.get_actions((board.state, player_idx))
        
        for action in actions:
            # Execute an action, create a board copy
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            # Update the player index
            new_player_idx = 1 if player_idx == 0 else 0
            
            # Find the minimum value using the maximum recursion call
            min_score = min(min_score, self.max_value(board_copy, new_player_idx, alpha, beta, depth - 1))
            
            # Early pruning
            if min_score <= alpha:
                return min_score
            beta = min(beta, min_score)
            
        return min_score

    def nega_max_search(self, board, player_idx, alpha, beta, depth):
        """
        Negamax search with alpha-beta pruning.
        
        Finds the best move for the player using the negamax algorithm,
        which is a variant of minimax that relies on the zero-sum property
        to simplify the implementation.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Search depth
            
        Returns:
            Tuple of (best action, score)
        """
        best_action = None
        max_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        
        # Iterate through all possible actions
        for action in actions:
            # Execute an action, create a board copy
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            # Finding the optimal score for the next level
            score = self.nega_max(board_copy, player_idx, alpha, beta, depth)
            if score > max_score:
                # Assign best action accordingly
                best_action = action
                max_score = score
                
        return best_action, max_score

    def nega_max(self, board, player_idx, alpha, beta, depth):
        """
        Helper function for negamax search.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Remaining search depth
            
        Returns:
            Best score for this node
        """
        if depth == 0 or board.is_termination_state():
            return self.evaluate_board(board, player_idx)
            
        # Iterate through all possible actions at the next level
        max_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        
        for action in actions:
            # Execute an action, create a board copy
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            # Update the player index
            new_player_idx = 1 if player_idx == 0 else 0
            
            # Negate the evaluation value (for negamax)
            score = -self.nega_max(board_copy, new_player_idx, -beta, -alpha, depth - 1)
            
            # Assign values and alpha
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            
            # Prune for efficiency
            if alpha >= beta:
                break
                
        return max_score

    def evaluate_board(self, board, player_idx):
        """
        Board evaluation function for minimax and negamax.
        
        Assigns a score to a board position based on various heuristics.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            
        Returns:
            Score for the board position from player_idx's perspective
        """
        # Assign the white player pieces, ball, and score
        white_score = 0
        WHITE_GOAL_ROW = 7
        white_players = board.state[0:5]
        white_ball = board.state[5]
        _, white_ball_y = board.decode_single_pos(white_ball)
        
        # Assign the black player pieces, ball, and score
        black_score = 0
        BLACK_GOAL_ROW = 0
        black_players = board.state[6:11]
        black_ball = board.state[11]
        _, black_ball_y = board.decode_single_pos(black_ball)
        
        # Iterate through white players and find a score
        for white_player in white_players:
            # Incentivize the white player moving upwards
            _, white_player_y = board.decode_single_pos(white_player)
            white_score += white_player_y
            
            # Incentivize the white player in a goal row
            if white_player_y == WHITE_GOAL_ROW:
                # Double score
                white_score += white_player_y
                # If ball is also in the goal row, incentivize this score further
                if white_ball_y == WHITE_GOAL_ROW:
                    white_score += white_ball_y
                    
        # Incentivize moving the white ball
        white_score += white_ball_y
        
        # Iterate through black players and find a score
        for black_player in black_players:
            # Incentivize the black player moving downwards
            _, black_player_y = board.decode_single_pos(black_player)
            # Offset by WHITE_GOAL_ROW (7) to make the score a positive value
            black_score += (WHITE_GOAL_ROW - black_player_y)
            
            # Incentivize the black player in a goal row
            if black_player_y == BLACK_GOAL_ROW:
                # Double score
                black_score += (WHITE_GOAL_ROW - black_player_y)
                # If ball is also in the goal row, incentivize this score further
                if black_ball_y == BLACK_GOAL_ROW:
                    black_score += (WHITE_GOAL_ROW - black_ball_y)
                    
        # Incentivize moving the black ball
        black_score += (WHITE_GOAL_ROW - black_ball_y)
        
        # Adjust white score by black score and vice versa
        new_white = white_score - black_score
        new_black = black_score - white_score
        
        # Incentivize moving the ball upwards (within a distance of two)
        new_white += white_ball_y if white_ball_y > 5 else 0
        new_black += (WHITE_GOAL_ROW - black_ball_y) if black_ball_y < 2 else 0
        
        # Heavily incentivize moving the ball into the goal row (WIN!)
        new_white += 100 if white_ball_y == WHITE_GOAL_ROW else 0
        new_black += 100 if black_ball_y == BLACK_GOAL_ROW else 0
        
        return new_white if player_idx == 0 else new_black
    
    def principle_variation_iterative_deepening(self, board, player_idx, depth):
        """
        Iterative Deepening Principal Variation Search.
        
        Uses iterative deepening to explore progressively deeper in the game
        tree, finding the best move using Principal Variation Search.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            depth: Maximum search depth
            
        Returns:
            Tuple of (best action, score)
        """
        best_action = None
        max_depth = depth
        cur_depth = 1
        max_score = -math.inf
        
        while cur_depth <= max_depth:
            # Search for a termination state at the current depth
            # We know a potential good action for our player exists at this depth
            # We will use principle variation search to find the best action
            action, score = self.principle_variation(board, player_idx, cur_depth, -math.inf, math.inf)
            if score > max_score:
                max_score = score
                best_action = action
            
            cur_depth += 1

        return best_action, max_score

    def principle_variation(self, board, player_idx, depth, alpha, beta):
        """
        Principal Variation Search algorithm entry point.
        
        A variant of alpha-beta pruning that attempts to minimize
        the number of full-window searches.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            depth: Search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Tuple of (best action, score)
        """
        best_action = None
        best_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        
        # Actions are a list of tuples. The first value in the tuple is the piece,
        # the second value is the encoded position of where the piece will move to
        
        for action in actions:
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            score = self.principle_variation_search(board_copy, player_idx, depth - 1, -math.inf, math.inf)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action, best_score

    def principle_variation_search(self, board, player_idx, depth, alpha, beta):
        """
        Principal Variation Search implementation.
        
        Args:
            board: Current BoardState
            player_idx: Player index (0 or 1)
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Best score for this node
        """
        if board.is_termination_state() or depth == 0:
            return self.evaluate_board(board, player_idx)
            
        actions = self.get_actions((board.state, player_idx))
        first_action = True
        
        # Sort actions by position to try promising moves first
        actions = list(actions)
        actions.sort(key=lambda x: x[1], reverse=True)
        
        for action in actions:
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            
            if first_action:
                # Full window search for the first action
                score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -beta, -alpha)
            else:
                # Null window search for remaining actions
                score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -alpha - 1, -alpha)
                if (score > alpha):
                    # Re-search with full window if null window search fails high
                    score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -beta, -alpha)
                    
            if score >= beta:
                return beta
                
            alpha = max(alpha, score)
            first_action = False
            
        return alpha
