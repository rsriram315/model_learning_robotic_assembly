from robosuite.models.objects import MujocoXMLObject


class PegObj(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            "./simulation/models/assets/objects/peg.xml",
            name=name,
            joints=None,
            # joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=True)
