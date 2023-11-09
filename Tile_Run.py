class TileRun(list):
    def __init__(self, tiles=None):
        super().__init__(tiles or [])

    def get_score(self):
        lowest_bound = self.get_lower_bound()
        if lowest_bound is None:
            return 30 * len(self)
        highest_bound = lowest_bound + len(self) - 1
        return (lowest_bound + highest_bound) * (highest_bound - lowest_bound + 1) // 2

    def get_lower_bound(self):
        lowest_bound = None
        index = 0
        for tile in self:
            if not tile.joker:
                lowest_bound = tile.number - index
                break
            index += 1
        return lowest_bound

    def get_upper_bound(self):
        lower_bound = self.get_lower_bound()
        if lower_bound is not None:
            return lower_bound + len(self) - 1
        return None

    def get_color(self):
        tile_set_color = None
        for tile in self:
            if not tile.joker:
                tile_set_color = tile.color
                break
        return tile_set_color

    def __eq__(self, other):
        if isinstance(other, TileRun):
            return len(self) == len(other) and all(tile in other for tile in self)
        return False
