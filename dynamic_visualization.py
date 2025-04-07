import open3d as o3d
import glob
import time
import os
from tqdm import tqdm
import numpy as np
import threading
import cv2
import re

class PointCloudPlayer:
    def __init__(self):
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.vis = None
        self.pcd = None
        self.ply_files = []
        self.frames = []
        self.geometry_added = False

    def natural_sort(self, l):
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key):
            return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def initialize_visualizer(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1920, height=1080)

        # Register keyboard callbacks
        self.vis.register_key_callback(32, self.pause_resume)
        self.vis.register_key_callback(27, self.stop)

        # Set rendering options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # White background
        opt.point_size = 2.0
        opt.show_coordinate_frame = True  # Show coordinate frame

    def setup_camera(self, pcd):
        """Set camera parameters"""
        view_control = self.vis.get_view_control()

        # Get the bounding box of the point cloud
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()

        # Calculate appropriate camera distance
        max_extent = np.max(extent)
        camera_distance = max_extent * 2.5

        # Set camera parameters
        view_control.set_zoom(0.7)
        view_control.set_front([0, 0, -1])  # Camera direction
        view_control.set_up([0, 1, 0])      # Up direction
        view_control.set_lookat(center)    # Look-at point (center of the point cloud)

        # Optionally adjust the following parameters
        view_control.change_field_of_view(step=60)  # Field of view

        # Update the view
        self.vis.poll_events()
        self.vis.update_renderer()

    def pause_resume(self, vis):
        self.is_paused = not self.is_paused
        return False

    def stop(self, vis):
        self.is_playing = False
        return False

    def update_point_cloud(self, new_pcd):
        """Update point cloud display"""
        if self.geometry_added:
            self.vis.remove_geometry(self.pcd)

        # Add color to the point cloud (optional)
        if not new_pcd.has_colors():
            new_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

        self.vis.add_geometry(new_pcd)
        self.geometry_added = True
        self.pcd = new_pcd

        # Update the view
        self.vis.poll_events()
        self.vis.update_renderer()

    def play(self, folder_path, fps=10, output_video_path=None):
        # Get all PLY files and sort them in natural order
        all_files = glob.glob(os.path.join(folder_path, "*.ply"))
        self.ply_files = self.natural_sort(all_files)
        print(f"Found {len(self.ply_files)} PLY files")

        if not self.ply_files:
            print("No PLY files found!")
            return

        self.initialize_visualizer()

        # Read the first point cloud file and set the initial view
        first_pcd = o3d.io.read_point_cloud(self.ply_files[0])
        if len(first_pcd.points) == 0:
            print("Warning: Point cloud is empty!")
            return

        self.update_point_cloud(first_pcd)
        self.setup_camera(first_pcd)  # Set camera parameters
        time.sleep(1)  # Wait for initialization

        self.is_playing = True
        frame_time = 1 / fps

        try:
            with tqdm(total=len(self.ply_files)) as pbar:
                for ply_file in self.ply_files:
                    if not self.is_playing:
                        break

                    while self.is_paused:
                        time.sleep(0.1)
                        self.vis.poll_events()
                        self.vis.update_renderer()

                    # Read and display the point cloud
                    new_pcd = o3d.io.read_point_cloud(ply_file)
                    if len(new_pcd.points) > 0:
                        self.update_point_cloud(new_pcd)
                        self.setup_camera(new_pcd)  # Reset camera parameters after updating the point cloud

                        # Save video if needed
                        if output_video_path:
                            self.frames.append(np.asarray(self.vis.capture_screen_float_buffer(False)))

                        pbar.update(1)
                        pbar.set_description(f"Processing: {os.path.basename(ply_file)}")

                    time.sleep(frame_time)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            if output_video_path and self.frames:
                self.save_video(output_video_path, fps)
            self.vis.destroy_window()

    def save_video(self, output_path, fps=30):
        if not self.frames:
            print("No frames available to generate video!")
            return

        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in tqdm(self.frames, desc="Saving video"):
            frame_rgb = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Video saved to: {output_path}")

    def play_with_controls(self, folder_path, fps=10, output_video_path=None):
        thread = threading.Thread(target=self.play, args=(folder_path, fps, output_video_path))
        thread.start()
        return thread

def main():
    folder_path = "" #your path of ply files
    output_video_path = "" #your output video path

    player = PointCloudPlayer()

    print(f"Starting point cloud sequence playback, folder path: {folder_path}")
    print(f"Output video path: {output_video_path}")

    play_thread = player.play_with_controls(
        folder_path,
        fps=10,
        output_video_path=output_video_path
    )
    play_thread.join()

if __name__ == "__main__":
    main()
