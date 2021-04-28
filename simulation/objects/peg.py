from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion 


class Peg(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("/home/paj7rng/amira_ML/simulation/assets/peg.xml"),
            name=name, joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=True)
