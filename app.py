__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 导入必要的库
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

prompt_template_qa="""直接回答问题，除非问题涉及结婚、生孩、工作、恋爱、买房这些内容时才参考使用内容后内容做出回答。
    
    问题: {question}
    内容：{context}
    回答:"""
def change_prompt1(*args):
    global prompt_template_qa 
    prompt_template_qa = """委婉认真地回答问题，除非问题涉及结婚、生孩、工作、恋爱、买房这些内容时才参考使用内容后内容做出回答。
    问题: {question}
    内容：{context}
    回答:"""

def change_prompt2(*args):
    global prompt_template_qa 
    prompt_template_qa = """转移话题地回答问题，除非问题涉及结婚、生孩、工作、恋爱、买房这些内容时才参考使用内容后内容做出回答。
    问题: {question}
    内容：{context}
    回答:"""

def change_prompt3(*args):
    global prompt_template_qa 
    prompt_template_qa = """阴阳怪气地回答问题，除非问题涉及结婚、生孩、工作、恋爱、买房这些内容时才参考使用内容后内容做出回答。
    问题: {question}
    内容：{context}
    回答:"""
    
def init():
    model_dir = snapshot_download('binbeing/SpringFestQA'
                                  , cache_dir='./')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 下载模型
    os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir sentence-transformer')


def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    llm = InternLM_LLM(model_path = "binbeing/SpringFestQA")

    # 定义一个 Prompt Template
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History: {chat_history}
    Follow Up Input: {question}
    Standalone question: """
    prompt_qg = PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"],
    )
    global prompt_template_qa 

    prompt_qa = PromptTemplate(
            template=prompt_template_qa, 
            input_variables=["context", "question"]
    )
    question_generator = LLMChain(llm=llm, prompt=prompt_qg)
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_qa)

    # 运行 chain
    qa_chain = ConversationalRetrievalChain(retriever=vectordb.as_retriever(),question_generator=question_generator,combine_docs_chain=doc_chain,)
    return qa_chain

class Model_center():
    """
    存储问答 Chain 的对象 
    """
    init()
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history:list):
        """
        调用问答链进行回答
        """
        chat_history_tuples = []
        #for message in chat_history:
            #chat_history_tuples.append((message[0], message[1]))
        chat_history_tuples = tuple(tuple(x) for x in chat_history)
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"question": question, "chat_history": chat_history_tuples})["answer"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>SpringFestQA</center></h1>
                <center>年关走亲访友渡劫助手</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
            chat_history=[]    
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        # 创建一个新的gr.Column，用于放置按钮。
        with gr.Column(scale=2):
            # 创建三个gr.Button组件，分别设置label参数为"类型1"，"类型2"和"类型3"，设置click参数为不同的函数，比如change_prompt1，change_prompt2和change_prompt3。
            type1_btn = gr.Button("委婉认真")
            type2_btn = gr.Button("转换话题")
            type3_btn = gr.Button("阴阳怪气")
            type1_btn.click(change_prompt1)
            type2_btn.click(change_prompt2)
            type3_btn.click(change_prompt3)
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
