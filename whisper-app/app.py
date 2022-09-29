import whisper
import os
import gradio as gr

import assets

model = whisper.load_model("small")


def inference(mic=None, audio_file=None):
    if mic is not None:
        audio = whisper.load_audio(mic)
    elif audio_file is not None:
        audio = whisper.load_audio(audio_file)
    else:
        print("input is none")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return result.text


title = "Whisper"

avail_models = whisper.available_models()
print("av model", avail_models)
b = gr.Blocks()

if __name__ == "__main__":
    with b:
        with gr.Column():
            gr.Markdown("### input")
            with gr.Row():
                audio = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="microphone",
                    type="filepath",
                    optional=True
                )

                audio_file = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="upload",
                    type="filepath",
                    optional=True
                )

            avail_models = whisper.available_models()
            model = gr.Checkboxgroup(avail_models, label="model")

            with gr.Accordion("settings", open=False):
                with_timestamp = gh.Checkbox(
                    label="with timestamp", value=True)

            button = gr.Button(label="transcribe")

        text = gr.Textbox(show_label=False, max_lines=10)
        button.click(inference, inputs=[audio, audio_file],
                     outputs=[text])
    b.launch()
