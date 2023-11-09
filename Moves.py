class Moves(list):
    def __init__(self):
        super().__init__()

    def set_player(self, player):
        self.player = player

    def __eq__(self, other):
        if isinstance(other, Moves):
            return self.same_moves(other)
        return False

    def same_moves(self, other):
        return len(self) == len(other) and self.same_move(other)

    def same_move(self, other):
        for move in self:
            if move not in other:
                return False
        return True
