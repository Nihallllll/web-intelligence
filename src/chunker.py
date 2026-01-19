from typing import List

def tokenizer(text:str) ->List[str]:
        return text.split()

def chunk_text(text :str,document_id :str , url :str , chunk_size =400,overlap=50):
        words = tokenizer(text)
        
        for word in words.__len__():
                