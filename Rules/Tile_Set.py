class TileSet(list):
    def __init__(self, tiles=None):
        super().__init__(tiles or [])

    def is_valid(self):
        return len(self) >= 3

    def get_score(self):
        pass  # This method is abstract and should be implemented in subclasses

    def contains_joker(self, joker):
        assert joker.joker
        for tile in self:
            if tile.joker and tile.color == joker.color:
                return True
        return False
