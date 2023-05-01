from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain import OpenAI

from langchain.vectorstores import Chroma
import gradio as gr

from langchain.chains.question_answering import load_qa_chain
qa_chain = load_qa_chain(OpenAI(temperature=0.7), chain_type="map_rerank")


import json

# create embedding from jsonloader 
path = "./source/blog.json"
# load json and get lines and slug
pages = []
slugs = []
def load_json(path=path):
    with open(path, "r") as f:
        raw = json.load(f)
    # get lines and slug
    for i, j in enumerate(raw["pages"]):
        j = json.loads(j)
        # flat to lines
        line = " ".join(j["lines"])
        if line == "":
            continue

        pages.append(line)
        s = {"source": j["slug"]}
        slugs.append(s)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = splitter.create_documents(texts=pages, metadatas=slugs)

    return docs


def split_documents(docs): 
    """documents を tokenize する

    Args:
        docs (_type_): _description_

    Returns:
        _type_: _description_
    """
    # split chauncs
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return splitter.split_documents(docs)


def get_embeddings_for_search(texts):
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings=embeddings, 
                                  #metadatas=[{"source": meta[i]} for i in range(len(meta))]
    )
    return docsearch

class Chain():
    def __init__(self, doc):
        self.doc = doc 
        self.chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=self.doc.as_retriever(), return_source_documents=True)

    def get(self, query):
        #res = self.chain.run(question=query)
        res = self.chain({"query": query})
        sources = []
        for r in res["source_documents"]:
            sources.append(f'https://uni-3.app/{r.metadata["source"]}')
        return res["result"], "\n".join(sources)


def main():
    # get docsearch
    docs = load_json()
    # get embeddings
    #print("docs", docs)
    docsearch = get_embeddings_for_search(docs)
    # init chain
    c = Chain(docsearch)

    # launch gradio app
    with gr.Blocks() as demo:
        # input query then get_query 
        input_query = gr.Textbox(label="Query")
        # show query result
        result = gr.Textbox(label="Result")
        source = gr.Markdown(label="source")

        b = gr.Button("run")
        b.click(c.get, inputs=[input_query], outputs=[result, source])

    demo.launch()
    

if __name__ == '__main__':
    main()
