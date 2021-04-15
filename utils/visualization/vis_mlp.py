from .base_vis import BaseVisualize


class MLPVisualize(BaseVisualize):
    def __init__(self, cfg, vis_dir="saved/visualizations"):
        super().__init__(cfg, vis_dir)
