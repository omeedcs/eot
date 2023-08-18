import numpy as np
import queue
import pytest
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem
from game import RandomPlayer, PassivePlayer, MiniMaxPlayer, IterativeDeepPVSPlayer, NegaMaxPlayer, PrincipleVariationPlayer

class TestSearch:

    def test_game_state_goal_state(self):
        b1 = BoardState()
        gsp = GameStateProblem(b1, b1, 0)

        sln = gsp.search_alg_fnc()
        ref = [(tuple((tuple(b1.state), 0)), None)]

        assert sln == ref

    ## NOTE: If you'd like to test multiple variants of your algorithms, enter their keys below
    ## in the parametrize function. Your set_search_alg should then set the correct method to
    ## use.
    @pytest.mark.parametrize("alg", ["", ""])
    def test_game_state_problem(self, alg):
        """
        Tests search based planning
        """
        b1 = BoardState()
        b2 = BoardState()
        b2.update(0, 14)

        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Single Step
        ref = [(tuple((tuple(b1.state), 0)), (0, 14)), (tuple((tuple(b2.state), 1)), None)]

        assert sln == ref

        b2 = BoardState()
        b2.update(0, 23)
        
        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Two Step:
        # (0, 14) or (0, 10) -> (any) -> (0, 23) -> (undo any) -> (None, goal state)

        # print(gsp.goal_state_set)
        # print(sln)
        # print("our sln", sln)
        assert len(sln) == 5 ## Player 1 needs to move once, then move the piece back

        assert sln[0] == (tuple((tuple(b1.state), 0)), (0, 14)) or sln[0] == (tuple((tuple(b1.state), 0)), (0, 10))
        assert sln[1][0][1] == 1
        assert sln[2][1] == (0, 23)
        assert sln[4] == (tuple((tuple(b2.state), 0)), None)

        # TODO OUR TESTS BELOW THIS LINE
        # our board tests
        initial_state = BoardState()
        goal_state = BoardState()

        goal_state.update(3, 19)
        goal_state.update(9, 38)
        goal_state.update(5, 19)
        goal_state.update(11, 38)

        gsp = GameStateProblem(initial_state, goal_state, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()
        assert len(sln) == 5
        
        # template for testing:

        # initial_state = BoardState()
        # goal_state = BoardState()

        # goal_state.update(3,19)
        # goal_state.update(9,38)
        # goal_state.update(5,19)
        # goal_state.update(7,46)
        # goal_state.update(2,16)
        # goal_state.update(7,41)
        # goal_state.update(2,31)
        # goal_state.update(11,41)
        # goal_state.update(5,31)
        # goal_state.update(9,25)

        # gsp = GameStateProblem(initial_state, goal_state, 0)
        # gsp.set_search_alg(alg)
        # sln = gsp.search_alg_fnc()
        # for sl in sln:
        #     print(sl)
        # assert False
        




    def test_initial_state(self):
        """
        Confirms the initial state of the game board
        """
        board = BoardState()
        assert board.decode_state == board.make_state()

        ref_state = [(1,0),(2,0),(3,0),(4,0),(5,0),(3,0),(1,7),(2,7),(3,7),(4,7),(5,7),(3,7)]

        assert board.decode_state == ref_state

    def test_generate_actions(self):
        sim = GameSimulator(None)
        generated_actions = sim.generate_valid_actions(0)
        assert (0,6) not in generated_actions
        assert (4,0) not in generated_actions

    ## NOTE: You are highly encouraged to add failing test cases here
    ## in order to test your validate_action implementation. To add an
    ## invalid action, fill in the action tuple, the player_idx, the
    ## validity boolean (would be False for invalid actions), and a
    ## unique portion of the descriptive error message that your raised
    ## ValueError should return. For example, if you raised:
    ## ValueError("Cannot divide by zero"), then you would pass some substring
    ## of that description for val_msg.
    @pytest.mark.parametrize("action,player,is_valid,val_msg", [

        # ((0,14), 0, True, ""),
        # ((0,16), 0, True, ""),
        # ((0,10), 0, True, ""),
        # ((5,1), 0, True, ""),
        # ((5,2), 0, True, ""),
        # ((5,4), 0, True, ""),
        # ((5,5), 0, True, ""),

        # # add failing test cases here. Our two error messages are:
        #     # Ball move is not valid.
        #     # Piece move is not valid.
        #     # 5 is ball move.
        
        # ((4,20), 0, True, ""),
        # ((4,18), 0, True, ""),
        # ((4,13), 0, False, "not valid"),
        # ((4,10), 0, True, ""),
        # ((4,11), 0, False, "not valid"),
        # ((4,4), 0, False, "not valid"),
        # ((4,6), 0, False, "not valid"),

        # black piece check
        ((1, 21), 1, True, ""),

        # testing more complex actions based on a new board state.
        ((5,41), 0, True, ""),
        ((5,15), 0, False, "Ball move is not valid."),
        ((5,17), 0, False, "Ball move is not valid."),
        ((5,27), 0, False, "Ball move is not valid."),
        ((5,36), 0, False, "Ball move is not valid."),
        ((5,39), 0, False, "Ball move is not valid."),
        ((5,50), 0, False, "Ball move is not valid."),
        ((5,54), 0, False, "Ball move is not valid."),
        ((5,25), 0, False, "Ball move is not valid."),
        ((5,30), 0, False, "Ball move is not valid."),
        ((5,39), 0, False, "Ball move is not valid."),

        # piece 15
        ((0, 28), 0, True, ""),
        ((0, 10), 0, True, ""),
        ((0, 24), 0, True, ""),
        ((0, 0), 0, True, ""),
        ((0, 30), 0, False, "Piece move is not valid."),

        # piece 17
        ((1, 30), 0, False, "Piece move is not valid."),
        ((1, 4), 0, True, ""),
        ((1, 32), 0, True, ""),
        ((1, 2), 0, True, ""),
        ((1, 26), 0, True, ""),

        # black ball
        ((5, 49), 1, True, ""),
        ((5, 25), 1, True, ""),
        ((5, 30), 1, True, ""),
        ((5, 54), 1, True, ""),
    ])

    def test_validate_action(self, action, player, is_valid, val_msg):
        sim = GameSimulator(None)
        bs = BoardState()
        bs.state = np.array([15,17,27,36,41,27,25,30,39,49,54,39])
        sim.game_state = bs
        if is_valid:
            assert sim.validate_action(action, player) == is_valid
        else:
            with pytest.raises(ValueError) as exinfo:
                result = sim.validate_action(action, player)
            assert val_msg in str(exinfo.value)

    @pytest.mark.parametrize("state,is_term", [
        ([1,2,3,4,5,3,50,51,52,53,54,52], False), ## Initial State
        ([1,2,3,4,5,55,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,54], True), ## Player 1 wins
        ([1,2,3,4,5,5,50,51,52,53,6,6], True), ## Player 2 wins
        ([1,2,3,4,5,5,50,4,52,53,6,6], False), ## Invalid State
    ])
    def test_termination_state(self, state, is_term):
        board = BoardState()
        board.state = np.array(state)
        board.decode_state = board.make_state()

        assert board.is_termination_state() == is_term

    def test_encoded_decode(self):
        board = BoardState()
        assert board.decode_state  == [board.decode_single_pos(x) for x in board.state]

        enc = np.array([board.encode_single_pos(x) for x in board.decode_state])
        assert np.all(enc == board.state)

    def test_is_valid(self):
        board = BoardState()
        assert board.is_valid()

        ## Out of bounds test
        board.update(0,-1)
        assert not board.is_valid()
        
        board.update(0,0)
        assert board.is_valid()
        
        ## Out of bounds test
        board.update(0,-1)
        board.update(6,56)
        assert not board.is_valid()
        
        ## Overlap test
        board.update(0,0)
        board.update(6,0)
        assert not board.is_valid()

        ## Ball is on index 0
        board.update(5,1)
        board.update(0,1)
        board.update(6,50)
        assert board.is_valid()

        ## Player is not holding the ball
        board.update(5,0)
        assert not board.is_valid()
        
        board.update(5,10)
        assert not board.is_valid()

    @pytest.mark.parametrize("state,reachable,player", [
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(0,1),(2,1),(1,2),(1,0)]),
            0
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(2,2)]),
            1
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(0,0)
            ],
            set(),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,3)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(1,2)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
    ]) 
    def test_ball_reachability(self, state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        assert predicted_reachable_encoded == encoded_reachable


    """
    ADVERSARIAL SEARCH TEST SUITE
    """

    @pytest.mark.parametrize("p1_class,p2_class,encoded_state_tuple,exp_winner,exp_stat", [ 

            # (PrincipleVariationPlayer, IterativeDeepPVSPlayer,
            # (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
            # "BLACK", "No issues"),

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # normal board configuration experimentation
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, MiniMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, IterativeDeepPVSPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),
        
        # # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, NegaMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, MiniMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, IterativeDeepPVSPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, NegaMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, MiniMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, IterativeDeepPVSPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, NegaMaxPlayer,
        #     (1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52),
        #     "WHITE", "No issues"),       

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # exact same as 9 tests above, but with a advantage to player 1
        # (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52)  
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, MiniMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, IterativeDeepPVSPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # MINIMAX AS PLAYER 1
        # (MiniMaxPlayer, NegaMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, MiniMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, IterativeDeepPVSPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # ID-PVS AS PLAYER 1
        # (IterativeDeepPVSPlayer, NegaMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, MiniMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, IterativeDeepPVSPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # # NEGAMAX AS PLAYER 1
        # (NegaMaxPlayer, NegaMaxPlayer,
        #     (29, 40, 3, 4, 5, 29, 28, 43, 52, 16, 54, 52),
        #     "WHITE", "No issues"),

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # exact same as 9 tests above, but with a disadvantage to player 1
        # (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23)
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # MINIMAX AS PLAYER 1
        (MiniMaxPlayer, MiniMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # MINIMAX AS PLAYER 1
        (MiniMaxPlayer, IterativeDeepPVSPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # MINIMAX AS PLAYER 1
        (MiniMaxPlayer, NegaMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # ID-PVS AS PLAYER 1
        (IterativeDeepPVSPlayer, MiniMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # ID-PVS AS PLAYER 1
        (IterativeDeepPVSPlayer, IterativeDeepPVSPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # ID-PVS AS PLAYER 1
        (IterativeDeepPVSPlayer, NegaMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # NEGAMAX AS PLAYER 1
        (NegaMaxPlayer, MiniMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # NEGAMAX AS PLAYER 1
        (NegaMaxPlayer, IterativeDeepPVSPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),

        # NEGAMAX AS PLAYER 1
        (NegaMaxPlayer, NegaMaxPlayer,
            (29, 28, 3, 4, 5, 29, 23, 51, 52, 38, 54, 23),
            "WHITE", "No issues"),


        



    ])

    # This function is the driver for our tests.
    # It leverages the pytest parameterization to test our adversarial search algorithms.
    # Other functions under this are to showcase what alternative things we used to test our algorithms.
    def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        
        # code used to extensively test our adversarial search algorithms. 
        # solely for extensive testing purposes.

        player_1_success = 0
        total_games = 25
        cur_game = 1
        while cur_game <= total_games:
            
            # simulate the game
            b1 = BoardState()
            b1.state = np.array(encoded_state_tuple) 
            b1.decode_state = b1.make_state()
            players = [
                p1_class(GameStateProblem(b1, b1, 0), 0), 
                p2_class(GameStateProblem(b1, b1, 0), 1)]
            
            sim = GameSimulator(players)
            sim.game_state = b1
            rounds, winner, status = sim.run()

            # check the result
            if winner == exp_winner:
                player_1_success += 1
            
            print("----------------------------------------------------")
            print("The winner of game ", cur_game, " was ", winner, ".")
            print("----------------------------------------------------")
            
            # move on to next game.
            cur_game += 1
        
        # find the success ratio for player 1
        success_ratio = player_1_success / total_games
         
        # format and print as a percentage
        print()
        print("----------------------------------------------------")
        print("This game was a total of 25 rounds.")
        print("Player #1 was ", p1_class, " and Player #2 was ", p2_class, ".")
        print("Player #1 won ", player_1_success, " out of ", total_games, " games.")
        print("Success Ratio: ", success_ratio * 100, "%")
        print("The starting board state was: ", encoded_state_tuple)
        print("----------------------------------------------------")
        print()

        # for logging purposes, we copy over the prints to a text file
        file_name = "success_ratio_matrix_testing_advantage"
        f = open(file_name, "a")
        stuff_to_write = "----------------------------------------------------\n"
        stuff_to_write += "This game was a total of 25 rounds.\n"
        stuff_to_write += "Player #1 was " + str(p1_class) + " and Player #2 was " + str(p2_class) + ".\n"
        stuff_to_write += "Player #1 won " + str(player_1_success) + " out of " + str(total_games) + " games.\n"
        stuff_to_write += "Success Ratio: " + str(success_ratio * 100) + "%\n"
        stuff_to_write += "The starting board state was: " + str(encoded_state_tuple) + "\n"
        stuff_to_write += "----------------------------------------------------\n"
        stuff_to_write += "\n"
        f.write(stuff_to_write)
        f.close()

        # driver code for extensive adversarial testing.

        # i = 0
        # while True:
        #     b1 = BoardState()
        #     b1.state = np.array(encoded_state_tuple) 
        #     b1.decode_state = b1.make_state()
        #     players = [
        #         p1_class(GameStateProblem(b1, b1, 0), 0), 
        #         p2_class(GameStateProblem(b1, b1, 0), 1)]
        #     sim = GameSimulator(players)
        #     sim.game_state = b1
        #     rounds, winner, status = sim.run()
        #     if winner != exp_winner:
        #         break
        #     else:
        #         i += 1
        #         print("----------------------------------------")
        #         print("GAME: ", i)
        #         print("WINNER: ", winner)
        #         print("STATUS: ", status)
        #         print("ROUNDS: ", rounds)
        #         print("----------------------------------------")
        # assert True

    """
    This function represents the code used to fill out the success ratio matrix for our assignment.
    We made the decision to run approximately 25 trials of every single possible combination of 
    algorithms.
    """
    def success_matrix_experimentation(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        player_1_success = 0
        total_games = 25
        cur_game = 0
        while cur_game < total_games:

            b1 = BoardState()
            b1.state = np.array(encoded_state_tuple) 
            b1.decode_state = b1.make_state()
            players = [
                p1_class(GameStateProblem(b1, b1, 0), 0), 
                p2_class(GameStateProblem(b1, b1, 0), 1)]
            
            sim = GameSimulator(players)
            sim.game_state = b1
            rounds, winner, status = sim.run()

            # in our case, the expected winner will always be player 1. this is how we
            # will keep track of the success ratio.
            if winner == exp_winner:
                player_1_success += 1
            cur_game += 1
        
        # find the success ratio for player 1
        success_ratio = player_1_success / total_games
         
        # format and print as a percentage
        print()
        print("----------------------------------------------------")
        print("This game was a total of 25 rounds.")
        print("Player #1 was ", p1_class, " and Player #2 was ", p2_class, ".")
        print("Player #1 won ", player_1_success, " out of ", total_games, " games.")
        print("Success Ratio: ", success_ratio * 100, "%")
        print("The starting board state was: ", encoded_state_tuple)
        print("----------------------------------------------------")
        print()


        # for logging purposes, we copy over the prints to a text file
        # must change to ensure we can organize the data properly across experiments.
        file_name = "success_ratio_matrix_disadvantage"
        f = open(file_name, "a")
        stuff_to_write = "----------------------------------------------------\n"
        stuff_to_write += "This game was a total of 25 rounds.\n"
        stuff_to_write += "Player #1 was " + str(p1_class) + " and Player #2 was " + str(p2_class) + ".\n"
        stuff_to_write += "Player #1 won " + str(player_1_success) + " out of " + str(total_games) + " games.\n"
        stuff_to_write += "Success Ratio: " + str(success_ratio * 100) + "%\n"
        stuff_to_write += "The starting board state was: " + str(encoded_state_tuple) + "\n"
        stuff_to_write += "----------------------------------------------------\n"
        stuff_to_write += "\n"
        f.write(stuff_to_write)
        f.close()
    


        
        
