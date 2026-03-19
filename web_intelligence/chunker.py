from typing import List
from dataclasses import dataclass
import re

@dataclass
class Chunk:
    text: str
    chunk_index: int
    word_count: int
    doc_id: str
    url: str


def split_into_sentences(text: str) -> List[str]:
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, document_id: str, url: str, 
               chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = count_words(sentence)
        
        if current_word_count + sentence_words > chunk_size and current_chunk_sentences:
            chunk_text_str = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                chunk_index=len(chunks),
                doc_id=document_id,
                url=url,
                word_count=current_word_count
            ))
            
            overlap_sentences = []
            overlap_count = 0
            
            for s in reversed(current_chunk_sentences):
                if overlap_count + count_words(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_count += count_words(s)
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_count
        
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_words
    
    if current_chunk_sentences:
        chunk_text_str = ' '.join(current_chunk_sentences)
        chunks.append(Chunk(
            text=chunk_text_str,
            chunk_index=len(chunks),
            doc_id=document_id,
            url=url,
            word_count=current_word_count
        ))
    
    return chunks


def tokenizer(text: str) -> List[str]:
    return text.split()