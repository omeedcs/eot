import numpy as np
import queue
from game import BoardState, GameSimulator, Rules
from queue import PriorityQueue
from math import dist
import math
import random

class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        The form of initial state is:
        ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.

        """
        self.search_alg_fnc = self.our_snazzy_search_algorithm

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    """ Here is an example:
    
    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
    """

    # in this method, we are returning a list of (state, action) pairs.
    # the first state and action in the list correspond to the initial state and action taken from
    # the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    # (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    def our_snazzy_search_algorithm(self):
        # Assign the start and goal states
        start_state = self.initial_state
        goal_state = self.goal_state_set
        # Create the frontier, parent, and cost
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
                # print(type(next_state))
                new_cost = cost[current_state] + 1
                # Check if the next state is the new best state
                if next_state not in cost or new_cost < cost[next_state]:
                    heuristic = new_cost
                    # TODO: Should we use vectorized distance here? Kinda slower...

                    # goal_state_config = np.array(list(goal_state)[0][0])
                    # new_state = np.array(next_state[0])
                    # dist = np.linalg.norm(goal_state_config - new_state)
                    # heuristic += dist

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
        while node != None:
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
        Given two arrays, find the first mismatched value and return the value and the index of the 
        mismatched value.

        Inputs: 
            next_state: is a array, which stores the next state
            prev_state: is a array, which stores the previous state
        Outputs:
            The list of mismatched values and the index of the mismatched values.

        """
        values = []
        idx = -1
        for i, (x, y) in enumerate(zip(next_state, prev_state)):
            if x != y:
                values = [x, y]
                idx = i
        return values, idx



    """
    ADVERSARIAL SEARCH ALGORITHMS + RANDOM PLAYER + PASSIVE PLAYER
    NOTE: Below this comment block encompasses all the code used in assignment #3 for PSR.
    """


    """
    A random player (a player that takes random actions).
    """
    def random_algorithm(self, enc_state):
        actions = self.get_actions(enc_state)
        curr_action = random.choice(actions)
        return (curr_action, 1)
    
    """
    A passive player (one that simply moves their ball around without moving any of their block pieces)
    """
    def passive_algorithm(self, enc_state):
        actions = self.get_actions(enc_state)
        ball_actions = []
        for action in actions:
            if action[0] == 5:
                ball_actions.append(action)
        curr_action = random.choice(ball_actions)
        return (curr_action, 1)

    """
    MINI-MAX WITH ALPHA-BETA PRUNING ALGORITHM 
    This algorithm is used to find the best move for the player by using the alpha-beta pruning algorithm.
    @Inputs:
        board: is a BoardState object, which stores the current state of the board
        player_idx: is an integer, which stores the player index
        alpha: is a float, which stores the alpha value
        beta: is a float, which stores the beta value
        depth: is an integer, which stores the depth of the search
    @Outputs:
        The best move for the player and the corresponding score.
    """
    def mini_max_search(self, board, player_idx, alpha, beta, depth):
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
            # Find the maxumum value using the minimum recursion call
            max_score = max(max_score, self.min_value(board_copy, new_player_idx, alpha, beta, depth - 1))
            # Early pruning
            if max_score >= beta:
                return max_score
            alpha = max(alpha, max_score)
        return max_score
    
    def min_value(self, board, player_idx, alpha, beta, depth):
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

    """
    NEGAMAX ALGORITHM
    This algorithm is used to find the best move for the player by using the negamax algorithm.
    It leverages alpha beta pruning to speed up the search.
    @Inputs:
        board: is a BoardState object, which stores the current state of the board
        player_idx: is an integer, which stores the player index
        alpha is an integer, which stores the alpha value
        beta is an integer, which stores the beta value
        depth: is an integer, which stores the depth of the search
    @Outputs:
        The best move for the player and the corresponding score.
    """
    def nega_max_search(self, board, player_idx, alpha, beta, depth):
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

    """
    ALPHA-BETA and NEGA-MAX SCORER
    This is purely a helper function for the minimax and negamax algorithms.
    @Inputs:
        board: is a BoardState object, which stores the current state of the board
        player_idx: is an integer, which stores the player index
    @Outputs:
        Correspondent score for the board configuration.
    """
    def evaluate_board(self, board, player_idx):
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
    
    
    # TODO: FIX ITERATIVE DEEPENING
    """
    ITERATIVE DEEPENING DEPTH-FIRST SEARCH WITH PRINCIPLE VARIATION AT OPTIMAL DEPTH
    This algorithm is used to find the best move for the player by using the iterative deepening algorithm.
    @Inputs:
        board: is a BoardState object, which stores the current state of the board
        player_idx: is an integer, which stores the player index
        depth: is an integer, which stores the depth of the search
    @Outputs:
        The best move for the player
    """
    def principle_variation_iterative_deepening(self, board, player_idx, depth):
        best_action = None
        max_depth = depth
        cur_depth = 1
        max_score = -math.inf
        while cur_depth <= max_depth:
            # search for a termination state at the current depth. 
                # we know a potential good action for our player exists at this depth
                # we will use principle variation search to find the best action
                action, score = self.principle_variation(board, player_idx, cur_depth, -math.inf, math.inf)
                if score > max_score:
                    max_score = score
                    best_action = action
                else: 
                    cur_depth += 1

        # no term state found, so we will use principle variation search to find the best action
        # using the max depth
        return best_action, max_score

    def principle_variation(self, board, player_idx, depth, alpha, beta):
        best_action = None
        best_score = -math.inf
        actions = self.get_actions((board.state, player_idx))
        # actions are a list of tuples. The first value in the tuple is the piece, the second value
        # is the encoded position of where the piece will move to
        # we want to sort the tuples by the second value, which is the encoded position, with
        # maximum value first
        
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

        if board.is_termination_state() or depth == 0:
            return self.evaluate_board(board, player_idx)
        actions = self.get_actions((board.state, player_idx))
        first_action = True
        actions.sort(key=lambda x: x[1], reverse=True)
        for action in actions:
            next_state = self.execute((board.state, player_idx), action)
            board_copy = BoardState()
            board_copy.state = next_state[0]
            board_copy.decode_state = board_copy.make_state()
            if first_action:
                score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -beta, -alpha)
            else:
                score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -alpha - 1, -alpha)
                if (score > alpha):
                    score = -self.principle_variation_search(board_copy, (player_idx + 1) % 2, depth - 1, -beta, -alpha)
            if score >= beta:
                return beta
            alpha = max(alpha, score)
            first_action = False
        return alpha
