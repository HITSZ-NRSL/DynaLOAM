import pickle
from pathlib import Path
from easydict import EasyDict

import numpy as np
from rd3d.utils import viz_utils
import threading
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class Gui:
    SETTING_3D_BACKGROUND = [0.5, 0.5, 0.5, 1]
    SETTING_PANEL_WIDTH = 15
    SETTING_SEQUENCE_RANGE_MIN = 0
    SETTING_SEQUENCE_RANGE_MAX = 0

    SETTING_POINTS_MATERIAL = rendering.MaterialRecord()
    SETTING_POINTS_MATERIAL.point_size = 2

    SETTING_BOXES_MATERIAL = rendering.MaterialRecord()
    SETTING_BOXES_MATERIAL.shader = "unlitLine"
    SETTING_BOXES_MATERIAL.line_width = 3

    def add_geometry(self, *args, **kwargs):
        self.invoke_times += 1
        # print(self.invoke_times)
        self._3d.scene.add_geometry(*args, **kwargs)

    def __init__(self):
        self.window = gui.Application.instance.create_window("Bank Visualization", 1920, 1080)
        self._3d = self.get_scene3d()
        self._settings_panel, self.start_slider, self.end_slider = self.get_setting_panel()
        self.invoke_times = 0

        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._3d)
        self.window.add_child(self._settings_panel)
        self.is_done = False
        self.previous_viz_range = None

    @property
    def current_visible_range(self):
        return self.start_slider.int_value, self.end_slider.int_value

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

    def get_setting_panel(self):
        em = self.window.theme.font_size
        vspace = int(round(0.5 * em))
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        viz_option_layout = gui.Horiz()
        viz_option_layout.add_child(gui.Checkbox("boxes"))
        viz_option_layout.add_child(gui.Checkbox("points"))
        layout.add_fixed(vspace)
        layout.add_child(viz_option_layout)

        """ sequence indices sliders """
        slider1_layout = gui.Horiz()
        slider1_layout.add_child(gui.Label("start"))
        slider1 = gui.Slider(gui.Slider.INT)
        slider1.set_limits(self.SETTING_SEQUENCE_RANGE_MIN, self.SETTING_SEQUENCE_RANGE_MAX)
        slider1.int_value = self.SETTING_SEQUENCE_RANGE_MIN
        slider1.set_on_value_changed(self._on_start_slider)
        slider1_layout.add_child(slider1)
        layout.add_fixed(vspace)
        layout.add_child(slider1_layout)

        slider2_layout = gui.Horiz()
        slider2_layout.add_child(gui.Label(" end "))
        slider2 = gui.Slider(gui.Slider.INT)
        slider2.int_value = self.SETTING_SEQUENCE_RANGE_MAX
        slider2.set_limits(self.SETTING_SEQUENCE_RANGE_MIN, self.SETTING_SEQUENCE_RANGE_MAX)
        slider2.set_on_value_changed(self._on_end_slider)
        slider2_layout.add_child(slider2)
        layout.add_fixed(vspace)
        layout.add_child(slider2_layout)

        return layout, slider1, slider2

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._3d.frame = r
        width = Gui.SETTING_PANEL_WIDTH * layout_context.theme.font_size
        height = self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height
        height = min(r.height, height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_start_slider(self, new_val):
        self.update_sliders()

    def _on_end_slider(self, new_val):
        self.update_sliders()

    def _on_key(self, e):
        if e.key == gui.KeyName.RIGHT and e.type == gui.KeyEvent.UP:
            self.is_update = False
            self.frame_idx += 1
            print("->")
            return gui.Widget.EventCallbackResult.HANDLED
        if e.key == gui.KeyName.LEFT and e.type == gui.KeyEvent.UP:
            self.is_update = False
            self.frame_idx -= 1
            print("<-")
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def update_sliders(self, begin=None, end=None):
        if begin or end:
            self.SETTING_SEQUENCE_RANGE_MIN = begin
            self.SETTING_SEQUENCE_RANGE_MAX = end
            self.end_slider.set_limits(begin, self.SETTING_SEQUENCE_RANGE_MAX)
            self.start_slider.set_limits(self.SETTING_SEQUENCE_RANGE_MIN, end)
            self.start_slider.int_value = begin
            self.end_slider.int_value = end
        else:
            self.end_slider.set_limits(self.start_slider.int_value, self.SETTING_SEQUENCE_RANGE_MAX)
            self.start_slider.set_limits(self.SETTING_SEQUENCE_RANGE_MIN, self.end_slider.int_value)
            self.update_visible()

    def set_visible(self, i, enable):
        self._3d.scene.show_geometry(f"points_{i}", enable)
        self._3d.scene.show_geometry(f"odom_{i}_0", enable)
        for tk_id in tracklets_each_frame[i]:
            self._3d.scene.show_geometry(tk_id, enable)

    def update_visible(self):
        p_begin, p_end = self.previous_viz_range
        begin, end = self.previous_viz_range = self.current_visible_range
        if p_begin <= begin:
            for i in range(p_begin, begin):
                self.set_visible(i, False)
        else:
            for i in range(begin, p_begin):
                self.set_visible(i, True)
        if p_end <= end:
            for i in range(p_end, end):
                self.set_visible(i, True)
        else:
            for i in range(end, p_end):
                self.set_visible(i, False)


class SequencesVisualizer(Gui):
    def __init__(self):
        super().__init__()

        self.frame_idx = 0
        threading.Thread(target=self._update_thread).start()

    def to_world_frame(self, b, t):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(t[:3, :3]).as_euler('xyz', degrees=False)
        b[:, :3] = (np.concatenate((b[:, :3], np.ones_like(b[..., :1])), axis=-1) @ t.T)[:, :3]
        b[:, 6] = np.arctan2(np.sin(b[:, 6] + rot[-1]), np.cos(b[:, 6] + rot[-1]))
        return b

    def draw_one_sequence_data(self):
        for i, (file, pose) in enumerate(zip(data_paths, poses)):
            points = np.fromfile(str(file), dtype=np.float32).reshape(-1, 4)[::10, :]
            xyz = np.concatenate((points[:, :3], np.ones_like(points[..., :1])), axis=-1)
            xyz = xyz @ pose.T
            points[:, :3] = xyz[:, :3]
            # viz_utils.add_points(vis=self, name=f"points_{i}",
            #                      points=points,
            #                      material=self.SETTING_POINTS_MATERIAL)
            # viz_utils.add_keypoint(vis=self, name=f"odom_{i}",
            #                        points=np.array([pose[:3, 3]]), radius=0.2, color=[1, 0, 0],
            #                        material=self.SETTING_POINTS_MATERIAL)
        print([len(t) for t in tracklets.values()])
        num_boxes = 0
        for i, tracklet in enumerate(tracklets[seq_name].values()):
            def color_map(x):
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('prism')
                return cmap(x / len(tracklets[seq_name]))[:3]

            if i > 80:
                print(num_boxes)
                break
            num_boxes += len(tracklet['boxes'])
            boxes = np.array(list(tracklet['refined_boxes'].values()))[:, :7]

            if 'motion_state' in tracklet and tracklet['motion_state'] == 0:
                colors = [0.5, 0.5, 0.5]
            else:
                colors = color_map(i)
            viz_utils.add_boxes(vis=self, name=f"track_{i}",
                                boxes=boxes, color=colors,
                                material=self.SETTING_BOXES_MATERIAL)
            # viz_utils.add_keypoint(vis=self._3d.scene, name=f"track_center_{i}",
            #                        points=np.array(tracklet['boxes'])[:, :3], radius=0.2, color=colors,
            #                        material=self.SETTING_POINTS_MATERIAL)
        for i, tk in enumerate(tracklets[seq_name].values()):
            for j, fid in enumerate(tk['frame_id']):
                tracklets_each_frame[int(fid.split('_')[-1])].append(f"track_{i}_{j}")

        self.update_sliders(0, len(data_paths))
        self.previous_viz_range = self.current_visible_range

    def _update_thread(self):
        import time
        gui.Application.instance.post_to_main_thread(self.window, self.draw_one_sequence_data)
        while not self.is_done:
            time.sleep(0.1)


if __name__ == '__main__':
    seq_name = "2011_09_28_drive_0038_sync"  # "2011_09_26_drive_0009_sync" "2011_09_28_drive_0038_sync"
    poses = np.load(f"/home/nrsl/workspace/temp/RobDet3D/data/custom/sequences/{seq_name}/pointclouds_poses.npy")
    data_paths = Path(f"/home/nrsl/workspace/temp/RobDet3D/data/custom/sequences/{seq_name}/pointclouds")
    data_paths = list(data_paths.iterdir())
    data_paths.sort()

    with open("tools/ms3d/data/devel/tracklets_raw_preds_1f_refine.pkl", 'rb') as f:
        tracklets = pickle.load(f)
        tracklets_each_frame = [[] for _ in range(len(data_paths) * 2)]
    # with open("tools/ms3d/data/devel/all_raw_preds_1f_fusion.pkl.bk", 'rb') as f:
    #     boxes = pickle.load(f)

    app = gui.Application.instance
    app.initialize()

    win = SequencesVisualizer()
    app.run()
