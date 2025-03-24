# StrategoSpheres

A strategic board game where players navigate their pieces and ball across the board to reach the opponent's goal line. Inspired by End of Track game.

## Game Overview

Stratego Spheres is a two-player turn-based strategy game implemented in Python. Each player controls five pieces and one ball, with the objective of moving their ball to the opponent's goal line. The game combines elements of chess and soccer, requiring both tactical positioning and strategic planning.

## Features

- Turn-based gameplay with alternating moves between two players
- Multiple AI player implementations with varying difficulty levels:
  - Random Player: Makes completely random valid moves
  - Passive Player: Only moves the ball, not the blocking pieces
  - MiniMax Player: Uses minimax algorithm with alpha-beta pruning
  - NegaMax Player: Implements the negamax variant of minimax
  - Principal Variation Search Player: Uses advanced search techniques for optimal moves
- Game simulation system that manages game state and enforces rules
- Extensible design allowing for easy implementation of additional AI strategies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stratego-spheres.git
cd stratego-spheres
```

2. No additional dependencies are required beyond Python standard libraries and NumPy.

3. Install NumPy if you don't have it already:
```bash
pip install numpy
```

## Usage

Run the game with two player types of your choice:

```python
from game import GameSimulator, RandomPlayer, MiniMaxPlayer

# Create players
player1 = RandomPlayer()
player2 = MiniMaxPlayer()

# Initialize game simulator with the players
game = GameSimulator(player1, player2)

# Run the game
winner = game.run()
print(f"Player {winner} wins!")
```

## Code Structure

The codebase is organized into two main files:

### game.py
- `BoardState`: Manages the game board representation and state
- `Rules`: Defines and enforces game rules and valid moves
- `GameSimulator`: Orchestrates gameplay between two players
- Player classes: Various player implementations with different strategies

### search.py
- `Problem`: Abstract base class for defining search problems
- `GameStateProblem`: Game-specific problem formulation
- Search algorithms: A*, Minimax, Negamax, and Principal Variation Search

## Game Rules

1. The game is played on an 8x8 board.
2. Each player has 5 pieces and 1 ball.
3. Players take turns moving one piece or their ball.
4. A piece can move one square in any direction (horizontally, vertically, or diagonally).
5. The ball can be moved to any reachable position (not blocked by other pieces).
6. The objective is to move your ball to the opponent's goal line.
7. The first player to reach the opponent's goal line with their ball wins.

## Contributing

Contributions to improve the game are welcome! Feel free to submit pull requests with enhancements or bug fixes.

## License

This project is open source and available under the [MIT License](LICENSE).