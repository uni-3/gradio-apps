import os
import gradio as gr
os.system("pip install git+https://github.com/openai/whisper.git")

import whisper

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
        with gr.Box():
            with gr.Row():
                audio = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="microphone",
                    type="filepath",
                    optional=True
                )

            with gr.Row():
                audio_file = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="upload",
                    type="filepath",
                    optional=True
                )
            button = gr.Button("transcribe")

        text = gr.Textbox(show_label=False)
        button.click(inference, inputs=[audio, audio_file],
                     outputs=[text])
    b.launch()
