class Game:
    def __init__(self):
        self.moves = []

    def is_end(self):
        return False

    def play_move(self, player, move):
        self.moves.append(move)

    def undo_move(self, player, move):
        self.moves.pop()

    def get_possible_moves(self, player):
        return []

    def evaluate(self, player):
        return 0.0

    def get_next_player(self, player):
        return player
