import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import torch

#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# https://qiita.com/DeepTama/items/7788aa518f349f0786d2

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

class ImageCaptioning:
    def __init__(self, device="cpu"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

    def inference(self, img, text):
        inputs = self.processor(img, text, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs, max_length=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        #print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions

# def inference(img, text):
#     inputs = processor(img, text, return_tensors="pt")

#     # unconditional image captioning
#     inputs = processor(img, return_tensors="pt")

#     out = model.generate(**inputs, max_length=50)
#     d = processor.decode(out[0], skip_special_tokens=True)
#     return d


def main():
    with gr.Blocks() as demo:
        img = gr.Image(image_mode="RGB", type="pil")
        text = gr.Textbox(lines=1, label="Text", value="a photography of")
        btn = gr.Button("Submit")
        output = gr.Textbox(lines=1, label="Output")

        imc = ImageCaptioning()
        btn.click(imc.inference, inputs=[img, text], outputs=[output])

        gr.Examples(
            examples=[img_url],inputs=[img],
            outputs=[output]
            )
    demo.launch()

if __name__ == "__main__":
    main()
