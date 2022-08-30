import gradio as gr
import numpy as np


def upload_img(input_img):
    return input_img


def echo(n):
    return "hello" + n
    
def res(x, y):
    """
    複数ファイルを受け取って処理して、1つのファイルを返すやつ
    """
    # return [np.fliplr(x), np.fliplr(y)]
    return np.fliplr(y)


demo = gr.Blocks()
if __name__ == "__main__":
    # demo = gr.Interface(fn=echo, inputs="text", outputs="text")
    # demo = gr.Interface(upload_img, gr.Image(shape=(200, 200)), "image")
    with demo:
        gr.Markdown("input src/target img")
        with gr.Row():
            src = gr.Image()
            target = gr.Image()

        img_output = gr.Image(label="result")
        img_button = gr.Button("show")

        img_button.click(res, inputs=[src, target],
                         outputs=img_output)
    demo.launch()
