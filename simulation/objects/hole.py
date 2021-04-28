from robosuite.models.objects import MujocoXMLObject


def test():
    pass


class RoundHole(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            "/home/paj7rng/amira_ML/simulation/assets/round-hole.xml",
            name=name,
            # joints=None,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object,
            also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic
