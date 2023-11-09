from typing import List

class Move:
    def __init__(self):
        self.heap = []  # Represents the "heap," which is a list of Tile objects
        self.from_tile_set = None  # Represents the TileSet from which the Move was played
        self.tiles = []  # List of Tile objects involved in the Move
        self.to_tile_sets = []  # List of TileSet objects resulting from the Move

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.same_move(other)
        return False

    def same_move(self, other):
        if self.from_tile_set is not other.from_tile_set or (self.from_tile_set is not None and self.from_tile_set != other.from_tile_set):
            return False
        if not all(tile in other.tiles for tile in self.tiles):
            return False
        for tile_set in self.to_tile_sets:
            if tile_set not in other.to_tile_sets:
                return False
        return True

    def __hash__(self):
        return super().__hash__()

    def add_tile(self, tile):
        self.tiles.append(tile)

    def add_to_tile_set(self, tile_set):
        self.to_tile_sets.append(tile_set)
