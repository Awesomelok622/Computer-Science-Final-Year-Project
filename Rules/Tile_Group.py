class TileGroup(list):
    def __init__(self, tiles=None):
        super().__init__(tiles or [])

    def get_score(self):
        score = 0
        joker_score = 0
        joker_count = 0
        for tile in self:
            if tile.joker:
                joker_count += 1
            else:
                score += tile.number
                joker_score = tile.number
        return score + joker_count * joker_score

    def __eq__(self, other):
        if isinstance(other, TileGroup):
            return len(self) == len(other) and all(tile in other for tile in self)
        return False
