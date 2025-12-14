import gradio as gr
from src.image_data_train_test_split.get_train_val_test_split import split_train_val_test

with gr.Blocks(title="YOLO Dataset Split Tools") as demo:
    gr.Markdown("## YOLO Dataset Train/Val/Test Split")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload & Settings")
            with gr.Group():
                with gr.Tab("ZIP File"):
                    zip_input = gr.File(
                        label="ZIP File (images + labels)",
                        file_types=[".zip"],
                        type="filepath",
                    )
                with gr.Tab("Directory Path"):
                    path_input = gr.Textbox(
                        label="Directory Path",
                        placeholder="/path/to/directory/with/images/and/labels",
                        info="Path to directory containing 'images' and 'labels' folders",
                    )
            
            # gr.Markdown("### Split Ratios")
            with gr.Row():
                with gr.Column(scale=1):
                    train_ratio_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Train Ratio",
                        info="Set to 0 to skip train split",
                    )
                    val_ratio_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.05,
                        label="Validation Ratio",
                        info="Set to 0 to skip validation split",
                    )
                    test_ratio_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                        label="Test Ratio",
                        info="Set to 0 to skip test split",
                    )
                    
                    seed_input = gr.Number(
                        value=42,
                        label="Random Seed",
                        info="Seed for reproducible splits",
                        precision=0,
                    )
                
            submit_btn = gr.Button("Create Split", variant="primary")
            

        with gr.Column(scale=1):
            gr.Markdown("### Split Information")
            text_output = gr.Textbox(
                label="Status",
                lines=10,
                interactive=False,
            )

            ratio_info = gr.Markdown(
                "**Current Ratios:**\n"
                "- Train: 70%\n"
                "- Val: 20%\n"
                "- Test: 10%\n\n"
                "Ratios must sum to 1.0. Set any ratio to 0 to skip that split."
            )

            download_btn = gr.File(
                label="Download Split Dataset (ZIP)",
                type="filepath",
                height=50,
            )
            
            with gr.Accordion("Instructions", open=False):
                gr.Markdown(
                    "1. Upload a ZIP containing `images` and `labels` **OR** provide a directory path\n"
                    "2. Adjust split ratios (must sum to 1.0)\n"
                    "3. Set random seed for reproducibility (optional)\n"
                    "4. Click **Create Split**\n"
                    "5. Download the split dataset as ZIP\n\n"
                    "**Output Structure:**\n"
                    "```\n"
                    "output/\n"
                    "├── train/\n"
                    "│   ├── images/\n"
                    "│   └── labels/\n"
                    "├── val/\n"
                    "│   ├── images/\n"
                    "│   └── labels/\n"
                    "└── test/\n"
                    "    ├── images/\n"
                    "    └── labels/\n"
                    "```\n\n"
                    "**Note:** Image-label pairs are kept together during splitting."
                )
            
            with gr.Accordion("Tips", open=False):
                gr.Markdown(
                    "- **Common splits:** 70/20/10, 80/20/0, 60/20/20\n"
                    "- You can skip any split by setting its ratio to 0\n"
                    "- The same seed will produce the same split\n"
                    "- All splits will include `classes.txt` if present in input"
                )

    def update_ratio_info(train, val, test):
        total = train + val + test
        status = "✅ Valid" if abs(total - 1.0) < 0.001 else "❌ Invalid"
        return f"**Current Ratios:**\n- Train: {train*100:.0f}%\n- Val: {val*100:.0f}%\n- Test: {test*100:.0f}%\n\n**Total: {total*100:.0f}%** {status}\n\nRatios must sum to 1.0."

    def process_split(zip_file, input_path, train_ratio, val_ratio, test_ratio, seed):
        output_dir, status_msg, zip_path = split_train_val_test(
            zip_file, input_path, train_ratio, val_ratio, test_ratio, seed
        )
        if output_dir is None:
            # Error occurred
            return status_msg, None
        return status_msg, zip_path

    # Update ratio info when sliders change
    train_ratio_slider.change(
        fn=update_ratio_info,
        inputs=[train_ratio_slider, val_ratio_slider, test_ratio_slider],
        outputs=ratio_info,
    )
    val_ratio_slider.change(
        fn=update_ratio_info,
        inputs=[train_ratio_slider, val_ratio_slider, test_ratio_slider],
        outputs=ratio_info,
    )
    test_ratio_slider.change(
        fn=update_ratio_info,
        inputs=[train_ratio_slider, val_ratio_slider, test_ratio_slider],
        outputs=ratio_info,
    )

    submit_btn.click(
        fn=process_split,
        inputs=[zip_input, path_input, train_ratio_slider, val_ratio_slider, test_ratio_slider, seed_input],
        outputs=[text_output, download_btn],
    )

if __name__ == "__main__":
    demo.launch(share=False)
