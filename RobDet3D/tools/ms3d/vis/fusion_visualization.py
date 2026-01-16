import pickle
import time
from pathlib import Path
from easydict import EasyDict

import numpy as np
from rd3d.utils import viz_utils
import threading
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def get_dataset():
    from rd3d.datasets.datasets.custom.custom_dataset import CustomDataset
    from rd3d.core.config import Config
    from rd3d import build_dataloader
    cfg = Config.fromfile("configs/base/datasets/custom.py")
    custom_dataset: CustomDataset = build_dataloader(cfg.DATASET, training=False)
    return custom_dataset


def color_map(self, x):
    import matplotlib.pyplot as plt
    return plt.get_cmap('tab20c')(x / len(all_boxes))[:3]


class Gui:
    SETTING_WINDOWS = ("All Boxes Visualization", int(1920 * 0.8), int(1080 * 0.8))
    SETTING_3D_BACKGROUND = [0.5, 0.5, 0.5, 1]
    SETTING_PANEL_WIDTH = 18
    SETTING_SEQUENCE_RANGE_MIN = 0
    SETTING_SEQUENCE_RANGE_MAX = 0

    SETTING_POINTS_MATERIAL = rendering.MaterialRecord()
    SETTING_POINTS_MATERIAL.point_size = 2

    SETTING_BOXES_MATERIAL = rendering.MaterialRecord()
    SETTING_BOXES_MATERIAL.shader = "unlitLine"
    SETTING_BOXES_MATERIAL.line_width = 3

    def __init__(self):
        self.window = gui.Application.instance.create_window(*self.SETTING_WINDOWS)
        self._3d = self.get_scene3d()
        self.checkboxes = self.get_checkbox()

        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)

        self.window.add_child(self._3d)
        self.window.add_child(self.checkboxes)

        self.is_done = False
        self.frame_idx = 0
        self.is_change_frame = True

    def get_checkbox(self):
        from functools import partial
        em = self.window.theme.font_size
        vspace = int(round(0.25 * em))
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        for k in all_boxes:
            name = k
            checkbox = gui.Checkbox(name)
            checkbox.checked = True
            checkbox.set_on_checked(partial(self._set_on_checked_visible, detector=k))
            layout.add_child(checkbox)
            layout.add_fixed(vspace)
        return layout

    def get_scene3d(self):
        s = gui.SceneWidget()
        s.scene = rendering.Open3DScene(self.window.renderer)
        s.scene.set_background(self.SETTING_3D_BACKGROUND)
        s.scene.show_axes(True)
        s.scene.scene.enable_sun_light(False)
        # self.widget3d.scene.view.set_post_processing(False)
        s.setup_camera(60, o3d.geometry.AxisAlignedBoundingBox([-0, -20, 0], [20, 20, 20]), [0, 0, 0])
        s.set_on_key(self._on_key)
        return s

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._3d.frame = r
        width = Gui.SETTING_PANEL_WIDTH * layout_context.theme.font_size
        height = self.checkboxes.calc_preferred_size(layout_context, gui.Widget.Constraints()).height
        height = min(r.height, height)
        self.checkboxes.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_key(self, e):
        if e.key == gui.KeyName.RIGHT and e.type == gui.KeyEvent.DOWN:
            self.frame_idx += 1
            self.frame_idx %= len(dataset)
            self.is_change_frame = True
            return gui.Widget.EventCallbackResult.HANDLED
        if e.key == gui.KeyName.LEFT and e.type == gui.KeyEvent.DOWN:
            self.frame_idx -= 1
            self.frame_idx %= len(dataset)
            self.is_change_frame = True
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED


class SequencesVisualizer(Gui):

    def __init__(self):
        super().__init__()
        self.visible_dict = {k: True for k in all_boxes}
        threading.Thread(target=self._update_thread).start()

    def change_visible(self, detector):
        for i in range(len(all_boxes[detector][self.frame_idx]['boxes_lidar'])):
            self._3d.scene.show_geometry(f"{detector}_boxes_{self.frame_idx}_{i}", self.visible_dict[detector])

    def update(self):
        info = dataset.infos[self.frame_idx]
        print(f"show {info['frame_id']}")

        points = dataset.get_lidar(info['sequence'], info['sample_idx'])

        def draw():
            self._3d.scene.clear_geometry()
            viz_utils.add_points(vis=self._3d.scene, name=f"point_{self.frame_idx}",
                                 points=points, material=self.SETTING_POINTS_MATERIAL)
            for k in all_boxes:
                viz_utils.add_boxes(vis=self._3d.scene, name=f"{k}_boxes_{self.frame_idx}",
                                    boxes=all_boxes[k][self.frame_idx]['boxes_lidar'][:, :7],
                                    material=self.SETTING_BOXES_MATERIAL)
                self.change_visible(k)

        gui.Application.instance.post_to_main_thread(self.window, draw)

    def _set_on_checked_visible(self, checked, detector):
        self.visible_dict[detector] = checked
        self.change_visible(detector)

    def _update_thread(self):
        while not self.is_done:
            if self.is_change_frame:
                self.update()
                self.is_change_frame = False
            else:
                time.sleep(0.1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    preds_path: Path = parser.parse_args().path
    all_boxes = {p.stem: pickle.load(open(p, 'rb')) for p in sorted(preds_path.glob("*.pkl"))}
    dataset = get_dataset()

    app = gui.Application.instance
    app.initialize()

    win = SequencesVisualizer()
    app.run()
