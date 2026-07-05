import gradio as gr
from src.image_data_plot import app as image_data_plot_app

with gr.Blocks(title="ML Utils") as demo:
    # -------- Sidebar --------
    with gr.Sidebar():
        gr.Markdown("## 🚀 ML Utils")
        gr.Markdown("---")
        gr.Markdown("### Navigation")
        home_btn = gr.Button("🏠 Home", variant="secondary")
        image_data_plot_btn = gr.Button("📊 Image Data Plot", variant="secondary")
        gr.Markdown("---")
        gr.Markdown("### Tools")
        gr.Markdown("- 📊 Image Data Plot")

    # -------- Main content using Tabs (hidden UI, controlled by buttons) --------
    with gr.Tabs(visible=False) as main_tabs:
        with gr.Tab("Home", id="home_tab"):
            gr.Markdown("# 🚀 ML Utils")
            gr.Markdown("*Machine Learning Utilities and Visualization Tools*")
            gr.Markdown("---")
            gr.Markdown("### Welcome!")
            gr.Markdown("Use the sidebar to navigate to different tools.")
            gr.Markdown("### Available Tools:")
            gr.Markdown("- **Image Data Plot**: Visualize YOLO detection and segmentation data")

        with gr.Tab("Image Data Plot", id="image_data_plot_tab"):
            image_data_plot_app.demo.render()

    # -------- Navigation wiring --------
    # Use gr.update() to change selected tab
    home_btn.click(
        fn=lambda: gr.update(selected="home_tab"),
        outputs=main_tabs
    )

    image_data_plot_btn.click(
        fn=lambda: gr.update(selected="image_data_plot_tab"),
        outputs=main_tabs
    )

if __name__ == "__main__":
    demo.launch(share=False)

