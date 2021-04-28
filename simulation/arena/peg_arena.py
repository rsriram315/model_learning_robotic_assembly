from robosuite.models.arenas import TableArena


class PegsArena(TableArena):
    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml="/home/paj7rng/amira_ML/simulation/assets/pegs_arena.xml",
        )

        # Get references to peg bodies
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
