from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self):
        print("model is loading...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self,text:str)->List[float]:
        """convert the text into vector"""
        vector = self.model.encode(text,convert_to_tensor=False)
        return vector.tolist()     

    def embed_batch(self, texts :List[str])->List[List[float]]:
        """Embedd multiple texts"""
        vector = self.model.encode(texts,convert_to_tensor=False)
        return vector.tolist()
