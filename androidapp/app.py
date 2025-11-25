import flet as ft
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# Add parent directory to path to import detection utilities
sys.path.append(str(Path(__file__).parent.parent))

# Import the barbell detection functionality
from barbell_processor import process_video, VideoProcessingResult
from visualization import create_path_visualization
import config


class BarPathApp:
    """Main application class for BarPath AI"""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self._setup_page()
        self._init_state()
        self._create_ui_components()
        self.build_ui()
    
    def _setup_page(self):
        """Configure page settings"""
        self.page.title = config.APP_TITLE
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 0
        self.page.scroll = ft.ScrollMode.AUTO
        self.page.bgcolor = ft.Colors.GREY_900
    
    def _init_state(self):
        """Initialize application state variables"""
        self.selected_video_path: Optional[str] = None
        self.processing_result: Optional[VideoProcessingResult] = None
        self.rotation_angle: int = 0
    
    def _create_ui_components(self):
        """Create all UI components"""
        # File picker
        self.video_picker = ft.FilePicker(on_result=self.on_video_selected)
        self.page.overlay.append(self.video_picker)
        
        # Status text
        self.status_text = ft.Text(
            "Select a video to start tracking",
            size=15,
            color=ft.Colors.GREY_400,
            weight=ft.FontWeight.W_400,
            text_align=ft.TextAlign.CENTER,
        )
        
        # Progress components
        self.progress_bar = ft.ProgressBar(
            width=400,
            visible=False,
            color=ft.Colors.BLUE_400,
            bgcolor=ft.Colors.BLUE_900,
            bar_height=6,
        )
        
        self.progress_text = ft.Text(
            "",
            size=13,
            visible=False,
            color=ft.Colors.BLUE_200
        )
        
        # Buttons
        self.select_video_button = self._create_button(
            "Select Video",
            ft.Icons.VIDEO_LIBRARY,
            self.pick_video,
            ft.Colors.BLUE_600,
            ft.Colors.BLUE_700
        )
        
        self.process_button = self._create_button(
            "Process Video",
            ft.Icons.PLAY_ARROW_ROUNDED,
            self.process_video_click,
            ft.Colors.GREEN_600,
            ft.Colors.GREEN_700,
            visible=False
        )
        
        # Rotation dropdown
        self.rotation_dropdown = ft.Dropdown(
            label="Rotate Video",
            options=[
                ft.dropdown.Option(key="0", text="No Rotation"),
                ft.dropdown.Option(key="90", text="90° Clockwise"),
                ft.dropdown.Option(key="-90", text="90° Counter-clockwise"),
            ],
            value="0",
            width=280,
            visible=False,
            on_change=self.on_rotation_change,
            bgcolor=ft.Colors.GREY_800,
            border_color=ft.Colors.GREY_700,
            border_radius=12,
            text_size=14,
        )
        
        # Results container
        self.results_container = ft.Column(
            visible=False,
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
        )
    
    def _create_button(self, text: str, icon, on_click, bg_color, hover_color, visible=True):
        """Helper method to create styled buttons"""
        return ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(icon, size=24),
                    ft.Text(text, size=16, weight=ft.FontWeight.W_500),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                on_click=on_click,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor={
                        ft.ControlState.DEFAULT: bg_color,
                        ft.ControlState.HOVERED: hover_color,
                    },
                    elevation={"pressed": 0, "": 2},
                    animation_duration=200,
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=20,
                ),
                height=56,
                width=280,
            ),
            visible=visible,
        )
    
    def build_ui(self):
        """Build the main UI layout"""
        main_content = ft.Column(
            [
                self._create_header(),
                self._create_selection_area(),
                self._create_progress_container(),
                self.results_container,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )
        self.page.add(main_content)
    
    def _create_header(self):
        """Create the application header"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(
                        ft.Icons.FITNESS_CENTER,
                        size=48,
                        color=ft.Colors.BLUE_400,
                    ),
                    margin=ft.margin.only(bottom=12),
                ),
                ft.Text(
                    "BarPath AI",
                    size=36,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE,
                ),
                ft.Container(height=8),
                ft.Text(
                    "AI-Powered Barbell Tracking",
                    size=15,
                    color=ft.Colors.GREY_400,
                    weight=ft.FontWeight.W_300,
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.only(top=40, bottom=40, left=20, right=20),
            alignment=ft.alignment.center,
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
                colors=[ft.Colors.GREY_900, ft.Colors.GREY_800],
            ),
        )
    
    def _create_selection_area(self):
        """Create the video selection area"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(
                        ft.Icons.CLOUD_UPLOAD_OUTLINED,
                        size=64,
                        color=ft.Colors.BLUE_300,
                    ),
                    margin=ft.margin.only(bottom=16),
                ),
                self.status_text,
                ft.Container(height=24),
                self.select_video_button,
                ft.Container(height=16),
                self.rotation_dropdown,
                ft.Container(height=16),
                self.process_button,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=40,
            margin=ft.margin.symmetric(horizontal=20, vertical=10),
            border_radius=16,
            bgcolor=ft.Colors.GREY_800,
            border=ft.border.all(1, ft.Colors.GREY_800),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                offset=ft.Offset(0, 4),
            ),
        )
    
    def _create_progress_container(self):
        """Create the progress indicator container"""
        self.progress_container = ft.Container(
            content=ft.Column([
                self.progress_text,
                ft.Container(height=12),
                self.progress_bar,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=40, vertical=24),
            margin=ft.margin.symmetric(horizontal=20, vertical=10),
            border_radius=16,
            bgcolor=ft.Colors.GREY_800,
            border=ft.border.all(1, ft.Colors.GREY_800),
            visible=False,
        )
        return self.progress_container
    
    # Event Handlers
    
    def pick_video(self, e):
        """Open file picker for video selection"""
        self.video_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=config.SUPPORTED_VIDEO_FORMATS,
            dialog_title="Select a video file"
        )
    
    def on_video_selected(self, e: ft.FilePickerResultEvent):
        """Handle video selection"""
        if not e.files or len(e.files) == 0:
            self._update_status("No video selected", ft.Colors.RED_400)
            self._hide_process_controls()
            return
        
        try:
            selected_file = e.files[0]
            
            if hasattr(selected_file, 'path') and selected_file.path:
                # Desktop mode
                self.selected_video_path = selected_file.path
                filename = os.path.basename(self.selected_video_path)
                self._update_status(f"Selected: {filename}", ft.Colors.GREEN_400)
                self._show_process_controls()
            else:
                # Web/mobile mode - not supported yet
                self._update_status("Please use desktop version for video processing", ft.Colors.ORANGE_400)
                self._hide_process_controls()
            
        except Exception as ex:
            self._update_status(f"Error: {str(ex)}", ft.Colors.RED_400)
            self._hide_process_controls()
    
    def on_rotation_change(self, e):
        """Handle rotation dropdown change"""
        self.rotation_angle = int(self.rotation_dropdown.value)
    
    def process_video_click(self, e):
        """Start video processing in a background thread"""
        if not self.selected_video_path:
            return
        
        self._set_processing_state(True)
        thread = threading.Thread(target=self.process_video_background)
        thread.start()
    
    def process_video_background(self):
        """Background thread for video processing"""
        try:
            result = process_video(
                video_path=self.selected_video_path,
                model_path=config.MODEL_PATH,
                output_dir=str(config.TEMP_DIR),
                progress_callback=self.update_progress,
                rotate=self.rotation_angle
            )
            
            self.processing_result = result
            self.on_processing_complete()
            
        except Exception as ex:
            self.on_processing_error(str(ex))
    
    def update_progress(self, progress: float, message: str):
        """Update progress bar (called from background thread)"""
        self.progress_bar.value = progress
        self.progress_text.value = message
        self.page.update()
    
    def on_processing_complete(self):
        """Handle successful video processing"""
        self._set_processing_state(False)
        self.display_results()
        self.page.update()
    
    def on_processing_error(self, error_message: str):
        """Handle processing error"""
        self._set_processing_state(False)
        self._update_status(f"Error: {error_message}", ft.Colors.RED_400)
        self.page.update()
    
    # UI State Management
    
    def _update_status(self, message: str, color):
        """Update status text"""
        self.status_text.value = message
        self.status_text.color = color
        self.page.update()
    
    def _show_process_controls(self):
        """Show processing controls"""
        self.process_button.visible = True
        self.rotation_dropdown.visible = True
        self.results_container.visible = False
        self.page.update()
    
    def _hide_process_controls(self):
        """Hide processing controls"""
        self.process_button.visible = False
        self.rotation_dropdown.visible = False
        self.page.update()
    
    def _set_processing_state(self, is_processing: bool):
        """Set UI state during processing"""
        self.select_video_button.disabled = is_processing
        self.process_button.disabled = is_processing
        self.progress_container.visible = is_processing
        self.progress_bar.visible = is_processing
        self.progress_text.visible = is_processing
        
        if is_processing:
            self.progress_text.value = "Initializing processing..."
            self.results_container.visible = False
        else:
            self.progress_container.visible = False
            self.progress_bar.visible = False
            self.progress_text.visible = False
    
    # Results Display
    
    def display_results(self):
        """Display the processing results"""
        if not self.processing_result:
            return
        
        result = self.processing_result
        self.results_container.controls.clear()
        
        # Build results UI
        self.results_container.controls.extend([
            self._create_results_header(),
            self._create_stats_row(result),
            ft.Container(height=12),
            self._create_video_section(result),
            self._create_path_section(result),
            ft.Container(height=40),
        ])
        
        self.results_container.visible = True
        self.page.update()
    
    def _create_results_header(self):
        """Create results header"""
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, size=32, color=ft.Colors.GREEN_400),
                ft.Text(
                    "Processing Complete!",
                    size=26,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE
                ),
            ], alignment=ft.MainAxisAlignment.CENTER),
            margin=ft.margin.only(top=20, bottom=30),
        )
    
    def _create_stats_row(self, result: VideoProcessingResult):
        """Create statistics row"""
        return ft.Row([
            self._create_stat_card(
                "Repetitions",
                str(result.repetition_count),
                ft.Icons.FITNESS_CENTER,
                ft.Colors.BLUE_400
            ),
            self._create_stat_card(
                "Total Frames",
                str(result.total_frames),
                ft.Icons.MOVIE_ROUNDED,
                ft.Colors.PURPLE_400
            ),
        ], alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=16)
    
    def _create_video_section(self, result: VideoProcessingResult):
        """Create video player section"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.VIDEO_FILE_ROUNDED, size=28, color=ft.Colors.BLUE_300),
                    ft.Text(
                        "Processed Video",
                        size=22,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.WHITE
                    ),
                ], spacing=12),
                ft.Container(height=16),
                ft.Container(
                    content=ft.Text(
                        os.path.basename(result.output_video_path),
                        size=13,
                        color=ft.Colors.GREY_400,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.padding.symmetric(horizontal=16),
                ),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Video(
                        playlist=[ft.VideoMedia(result.output_video_path)],
                        playlist_mode=ft.PlaylistMode.LOOP,
                        fill_color=ft.Colors.BLACK,
                        aspect_ratio=16/9,
                        volume=100,
                        autoplay=False,
                        show_controls=True,
                        filter_quality=ft.FilterQuality.HIGH,
                    ),
                    width=min(700, self.page.width - 100) if hasattr(self.page, 'width') else 700,
                    border_radius=12,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    border=ft.border.all(2, ft.Colors.GREY_700),
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=28,
            margin=ft.margin.symmetric(horizontal=20, vertical=12),
            border_radius=16,
            bgcolor=ft.Colors.GREY_800,
            border=ft.border.all(1, ft.Colors.GREY_800),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )
    
    def _create_path_section(self, result: VideoProcessingResult):
        """Create bar path visualization section"""
        path_image_path = None
        if result.averaged_down_phase or result.averaged_up_phase:
            path_image_path = create_path_visualization(
                result.averaged_down_phase,
                result.averaged_up_phase,
                result.video_dimensions,
                output_path=os.path.join(str(config.TEMP_DIR), "path_visualization.png")
            )
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SHOW_CHART_ROUNDED, size=28, color=ft.Colors.GREEN_300),
                    ft.Text(
                        "Average Bar Path",
                        size=22,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.WHITE
                    ),
                ], spacing=12),
                ft.Container(height=12),
                ft.Text(
                    "Averaged path across all repetitions",
                    size=13,
                    color=ft.Colors.GREY_400,
                    italic=True,
                ),
                ft.Container(height=20),
                ft.Image(
                    src=path_image_path if path_image_path else "",
                    width=min(600, self.page.width - 100) if hasattr(self.page, 'width') else 600,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10,
                ) if path_image_path else ft.Text(
                    "No path visualization available",
                    size=14,
                    color=ft.Colors.GREY_400
                ),
                ft.Container(height=16),
                self._create_legend(),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=28,
            margin=ft.margin.symmetric(horizontal=20, vertical=12),
            border_radius=16,
            bgcolor=ft.Colors.GREY_800,
            border=ft.border.all(1, ft.Colors.GREY_800),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )
    
    def _create_legend(self):
        """Create legend for path visualization"""
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            width=4,
                            height=16,
                            bgcolor=ft.Colors.GREEN_400,
                            border_radius=2,
                        ),
                        ft.Text("Downward", size=12, color=ft.Colors.GREY_300, weight=ft.FontWeight.W_500),
                    ], spacing=8),
                    padding=8,
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREEN_400),
                    border_radius=8,
                ),
                ft.Container(width=12),
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            width=4,
                            height=16,
                            bgcolor=ft.Colors.RED_400,
                            border_radius=2,
                        ),
                        ft.Text("Upward", size=12, color=ft.Colors.GREY_300, weight=ft.FontWeight.W_500),
                    ], spacing=8),
                    padding=8,
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.RED_400),
                    border_radius=8,
                ),
            ], alignment=ft.MainAxisAlignment.CENTER),
        )
    
    def _create_stat_card(self, label: str, value: str, icon, color):
        """Create a statistic card"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(icon, size=36, color=color),
                    padding=12,
                    bgcolor=ft.Colors.with_opacity(0.15, color),
                    border_radius=12,
                ),
                ft.Container(height=12),
                ft.Text(value, size=36, weight=ft.FontWeight.BOLD, color=color),
                ft.Container(height=4),
                ft.Text(label, size=13, color=ft.Colors.GREY_400, weight=ft.FontWeight.W_400),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=24,
            margin=8,
            border_radius=16,
            bgcolor=ft.Colors.GREY_800,
            border=ft.border.all(1, ft.Colors.GREY_800),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
            width=160,
        )


def main(page: ft.Page):
    app = BarPathApp(page)


ft.app(target=main)