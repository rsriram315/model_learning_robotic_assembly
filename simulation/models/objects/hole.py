from robosuite.models.objects import MujocoXMLObject


class RoundHole(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            "./simulation/models/assets/objects/round-hole.xml",
            # "./simulation/models/assets/objects/decomposed_bosch_hole.xml",
            name=name,
            joints=None,
            # joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=True)
