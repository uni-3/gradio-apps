import gradio as gr

import os
os.system("pip install git+https://github.com/openai/whisper.git")
import gradio as gr

import whisper

model = whisper.load_model("small")


def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return result.text


title = "Whisper"

description = "Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification."

b = gr.Blocks()

if __name__ == "__main__":
    # demo = gr.Interface(fn=echo, inputs="text", outputs="text")
    # demo = gr.Interface(upload_img, gr.Image(shape=(200, 200)), "image")
    with b:
        gr.Markdown("input src/target img")
        with gr.Row():
            src = gr.Image()
            target = gr.Image()

        img_output = gr.Image(label="result")
        img_button = gr.Button("show")

        img_button.click(res, inputs=[src, target],
                         outputs=img_output)
    demo.launch()
