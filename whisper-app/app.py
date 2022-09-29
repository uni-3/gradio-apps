import whisper
import os
import gradio as gr


def inference(mic, audio_file, model_name, without_timestamp):
    model = whisper.load_model(model_name)

    if mic is not None:
        audio = whisper.load_audio(mic)
    elif audio_file is not None:
        audio = whisper.load_audio(audio_file)
    else:
        print("input is none")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(
        fp16=False, without_timestamps=without_timestamp,
        language="ja")
    result = whisper.decode(model, mel, options)

    print(result.get("segments"))
    print(result.text)
    return result.text


title = "Whisper"

avail_models = whisper.available_models()
# 日本語のみ対応のため
avail_models = [m for m in avail_models if "en" not in m]
b = gr.Blocks()

if __name__ == "__main__":
    with b:
        with gr.Column():
            gr.Markdown("### input 日本語音声のみ")
            with gr.Row():
                audio = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="microphone",
                    type="filepath",
                )

                audio_file = gr.Audio(
                    label="input audio",
                    show_label=False,
                    source="upload",
                    type="filepath",
                )

            model = gr.Radio(avail_models, label="model")

            without_timestamp = gr.Checkbox(
                label="without timestamp", value=False)

            button = gr.Button(label="transcribe")

        text = gr.Textbox(show_label=False, max_lines=10)
        button.click(fn=inference, inputs=[audio, audio_file, model, without_timestamp],
                     outputs=[text])
    b.launch()
