import dataclasses
from typing import List

@dataclasses.dataclass
class GameState:
    drawn_tile_indexes: List[int]
    final_table: Table
    final_pool: Pool
    final_players: List[Player]
