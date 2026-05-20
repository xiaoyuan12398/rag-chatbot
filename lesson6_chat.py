import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 页面配置
st.set_page_config(
    page_title="RAG文档问答助手",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG文档问答助手")

# ---------------------- 工具函数 ----------------------
def extract_text_from_file(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except:
        pass
    return text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def init_vector_db(api_key):
    api_key = api_key.strip()
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="rag_docs"
    )
    return vectorstore

def add_documents_to_db(vectorstore, chunks):
    try:
        ids = vectorstore.get()["ids"]
        if ids:
            vectorstore.delete(ids=ids)
    except:
        pass
    vectorstore.add_texts(chunks)

def query_vector_db(vectorstore, query, n_results=3):
    docs = vectorstore.similarity_search(query, k=n_results)
    return [doc.page_content for doc in docs]

# ---------------------- 侧边栏 ----------------------
with st.sidebar:
    st.header("配置")
    api_key = st.text_input("通义千问API密钥", type="password")
    model_name = st.selectbox("选择模型", ["qwen-turbo", "qwen-plus", "qwen-max"], index=0)
    temperature = st.slider("温度", 0.0, 1.0, 0.1)
    max_tokens = st.number_input("最大生成长度", 100, 4096, 2048)

    st.divider()
    st.header("上传文档")
    uploaded_file = st.file_uploader("上传PDF / Word", type=["pdf", "docx"])

    if uploaded_file and api_key:
        with st.spinner("处理中..."):
            text = extract_text_from_file(uploaded_file)
            chunks = split_text_into_chunks(text)
            vs = init_vector_db(api_key)
            add_documents_to_db(vs, chunks)
            st.success(f"完成！{len(chunks)} 块")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("停止生成"):
            st.session_state.stop = True

# ---------------------- 会话状态 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop" not in st.session_state:
    st.session_state.stop = False

# ---------------------- 显示历史 ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------- 问答 ----------------------
q = st.chat_input("输入问题...")
if q:
    if not api_key:
        st.error("请输入API密钥")
        st.stop()

    st.session_state.stop = False
    client = OpenAI(api_key=api_key.strip(), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 检索
    context = ""
    if uploaded_file:
        vs = init_vector_db(api_key)
        chunks = query_vector_db(vs, q)
        context = "\n\n".join(chunks)

    # 提示词
    if context:
        system = f"基于文档回答，没有就说没有。\n{context}"
    else:
        system = "你是助手"

    messages = [{"role": "system", "content": system}] + st.session_state.messages + [{"role": "user", "content": q}]
    st.session_state.messages.append({"role": "user", "content": q})

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 思考中...")
        full = ""

        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            placeholder.empty()
            ans = st.empty()

            for chunk in res:
                if st.session_state.stop:
                    break
                if chunk.choices[0].delta.content:
                    full += chunk.choices[0].delta.content
                    ans.markdown(full + "▌")

            ans.markdown(full)
            st.session_state.messages.append({"role": "assistant", "content": full})

        except Exception as e:
            st.error(f"失败：{str(e)}")