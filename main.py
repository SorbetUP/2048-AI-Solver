from game import Game
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board, score):
    clear()
    print("-" * 25)
    for row in board.grid:
        print("|", end="")
        for val in row:
            if val == 0:
                print("    .", end=" ")
            else:
                print(f"{val:5}", end=" ")
        print("|")
    print("-" * 25)
    print(f"Score : {score}")
    print("Utilisez z (up), s (down), q (left), d (right) pour jouer. 'x' pour quitter.")

def main():
    game = Game()
    while not game.is_over():
        print_board(game.board, game.score)
        move = input("Direction (z/s/q/d): ").lower()
        if move == 'x':
            print("Partie terminée.")
            break
        mapping = {'z': 'up', 's': 'down', 'q': 'left', 'd': 'right'}
        if move not in mapping:
            print("Commande invalide.")
            continue
        moved, gained = game.move(mapping[move])
        if not moved:
            print("Déplacement impossible, essaie autre chose.")
    else:
        print_board(game.board, game.score)
        if game.is_won():
            print("Bravo, tu as gagné !")
        else:
            print("Game over.")

if __name__ == "__main__":
    main()
