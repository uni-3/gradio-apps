import whisper
import gradio as gr

import os
os.system("pip install git+https://github.com/openai/whisper.git")


model = whisper.load_model("small")


def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    l, probs = model.detect_language(mel)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print(l, probs)
    print(result.text)
    return result.text


title = "Whisper"

b = gr.Blocks()

if __name__ == "__main__":
    with b:
        gr.Markdown("input audio")
        with gr.Group():
            with gr.Box():
                with gr.Row():
                    audio = gr.Audio(
                        label="input audio",
                        show_label=False,
                        source="microphone",
                        type="filepath"
                    )

                button = gr.Button("transcribe")

            text = gr.TextBox(show_label=False)
            button.click(inference, inputs=[audio],
                         outputs=[text])
    b.launch()
