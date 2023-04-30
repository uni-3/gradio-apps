from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
import gradio as gr



list_image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Hyla_japonica_sep01.jpg/260px-Hyla_japonica_sep01.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Tibur%C3%B3n_azul_%28Prionace_glauca%29%2C_canal_Fayal-Pico%2C_islas_Azores%2C_Portugal%2C_2020-07-27%2C_DD_14.jpg/270px-Tibur%C3%B3n_azul_%28Prionace_glauca%29%2C_canal_Fayal-Pico%2C_islas_Azores%2C_Portugal%2C_2020-07-27%2C_DD_14.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Thure_de_Thulstrup_-_Battle_of_Shiloh.jpg/251px-Thure_de_Thulstrup_-_Battle_of_Shiloh.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Passion_fruits_-_whole_and_halved.jpg/270px-Passion_fruits_-_whole_and_halved.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Messier83_-_Heic1403a.jpg/277px-Messier83_-_Heic1403a.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/2022-01-22_Men%27s_World_Cup_at_2021-22_St._Moritz%E2%80%93Celerina_Luge_World_Cup_and_European_Championships_by_Sandro_Halank%E2%80%93257.jpg/288px-2022-01-22_Men%27s_World_Cup_at_2021-22_St._Moritz%E2%80%93Celerina_Luge_World_Cup_and_European_Championships_by_Sandro_Halank%E2%80%93257.jpg',
    #'https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Wiesen_Pippau_%28Crepis_biennis%29-20220624-RM-123950.jpg/224px-Wiesen_Pippau_%28Crepis_biennis%29-20220624-RM-123950.jpg',
]
def get_doc():
    loader = ImageCaptionLoader(path_images=list_image_urls)
    return loader

class Query():
    def __init__(self, index):
        self.index = index 

    def get_query(self, query):
        return self.index.query(query)

def main():
    query = "What kind of images are there?"
    loader = get_doc()
    list_docs = loader.load()
    index = VectorstoreIndexCreator().from_loaders([loader])
    q = Query(index)

    # launch gradio app
    with gr.Blocks() as demo:
        # show images
        gr.Markdown("## Images")
        #for i in range(len(list_image_urls)):
        #    img = gr.Image(label=f"Image {i}")
        #    img.image = list_image_urls[i]
        # show list docs in textbox
        gr.Markdown("## list docs")
        for doc in list_docs:
            gr.Markdown(doc.page_content)

        gr.Markdown("## image caption and query")
        # input query then get_query 
        input_query = gr.Textbox(label="Query", value=query)
        # show query result
        gr.Markdown("## Results")
        query_result = gr.Textbox(label="Query Result")

        b = gr.Button("run")
        b.click(q.get_query, inputs=[input_query], outputs=[query_result])

    demo.launch()
    

if __name__ == '__main__':
    main()
