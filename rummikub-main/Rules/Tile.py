class Tile:
    def __init__(self, number=None, color=None, joker=False):
        self.number = number
        self.color = color
        self.joker = joker

    def __str__(self):
        if not self.joker:
            return f"{self.number}/{self.color}"
        else:
            return f"JOKER/{self.color}"

    def __eq__(self, other):
        if isinstance(other, Tile):
            return self.same_joker(other) or (self.same_number(other) and self.same_color(other))
        return False

    def same_joker(self, other):
        return self.joker and other.joker and self.same_color(other)

    def same_color(self, other):
        return self.color == other.color

    def same_number(self, other):
        return self.number == other.number

# Define TileColor enumeration if not defined elsewhere
class TileColor:
    # You may define the colors here if they are not predefined elsewhere
    pass
