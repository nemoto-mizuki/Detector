import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
import time


    
class Visualizer():
    def __init__(self):
        self.pcd_data = o3d.geometry.PointCloud()
        self.window = gui.Application.instance.create_window(
            "Add Spheres Example", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0, 0, 0, 0])
        bounds = self.scene.scene.bounding_box
        self.rgb_widget = gui.ImageWidget(o3d.geometry.Image(self.rgb_image))
        self.show_current = o3d.geometry.PointCloud()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.window.add_child(self.scene)      
        self.is_done = False
        threading.Thread(target=self.main_loop).start()
    
    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = contentRect.width // 2   # 15 ems wide
        self.scene.frame = gui.Rect(contentRect.x, contentRect.y,
                                      panel_width,
                                       contentRect.height)
        self.rgb_widget.frame = gui.Rect(panel_width,
                                    contentRect.y,contentRect.width -  panel_width,
                                    contentRect.height)
    
    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close