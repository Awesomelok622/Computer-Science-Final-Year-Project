import argparse
from rummikub.service import PlayerController  # Assuming PlayerController is in a service module

def main():
    parser = argparse.ArgumentParser(description="Rummikub Game")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to play")

    args = parser.parse_args()

    player_controller = PlayerController()
    player_controller.play(args.rounds)

if __name__ == "__main__":
    main()
