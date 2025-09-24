from sentence_transformers import SentenceTransformer
from typing import List
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

class M3eEmbeddings(Embeddings):
    def __init__(self):
        # 加载嵌入模型（如您之前的 m3e-base）
        # TODO: path of embedding model -> config
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))  # M3eEmbedding.py 所在目录
        model_path = os.path.join(base_dir, "m3e-base")       # 拼接出 app/models/m3e-base
        self.model = SentenceTransformer(model_path)
        # self.model = SentenceTransformer("./models/m3e-base")

    def embed_query(self, text: str) -> List[float]:
        """
        实现 LangChain 所需的 embed_query 方法，返回查询文本的嵌入向量；
        此处的text是string，返回List[float]
        """
        embedding = self.model.encode(text)
        return embedding.tolist()  # Convert numpy array to list

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        实现 LangChain 所需的 embed_documents 方法，返回文档的嵌入向量
        """
        embeddings = self.model.encode(documents)  # 使用 encode 方法生成嵌入向量
        return embeddings.tolist()  # Convert numpy array to list of lists

    def embed_text(self, text):
        """
        实现 LangChain 所需的 embed_text 方法，返回单个文本的嵌入向量；
        此处的text是string
        """
        embedding = self.model.encode(text)
        return embedding.tolist()  # Convert numpy array to list

    def __call__(self, text):
        """
        实现 LangChain 所需的 __call__ 方法，使类实例可调用；
        当 FAISS 调用 embedding_function(text) 时会使用此方法
        """
        return self.embed_text(text)