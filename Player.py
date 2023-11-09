class Player:
    def __init__(self, name):
        self.name = name
        self.rack = Rack()
        self.started = False

    def is_finished(self):
        return self.rack.is_empty()

    def add_tile_to_rack(self, tile):
        self.rack.add(tile)

    def remove_tile_from_rack(self, tile):
        self.rack.remove(tile)

    def remove_all_tiles_from_rack(self, tiles):
        for tile in tiles:
            self.rack.remove(tile)

class Rack(list):
    def is_empty(self):
        return not bool(self)

class Tile:
    def __init__(self):
        # You may need to define the properties of a Tile here.
        pass
