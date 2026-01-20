from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
        text:str
        chunk_index: int
        token_count :int
        doc_id :str
        url :str

def tokenizer(text:str) ->List[str]:
        return text.split()

def chunk_text(text :str,document_id :str , url :str , chunk_size :int =400,overlap:int =50)->List[Chunk]:
        words = tokenizer(text)
        chunks=[]
        for i in range(0, len(words), chunk_size - overlap ):
                chunk_token = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_token)
                
                chunks.append(Chunk(
                        text=chunk_text,
                        chunk_index =len(chunks),
                        doc_id=document_id,
                        url=url,
                        token_count=len(chunk_token)
                ))

        return chunks



                