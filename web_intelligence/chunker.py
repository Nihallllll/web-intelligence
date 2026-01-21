from typing import List
from dataclasses import dataclass
import re

@dataclass
class Chunk:
        text: str
        chunk_index: int
        token_count: int
        doc_id: str
        url: str


def split_into_sentences(text: str) -> List[str]:
    """
    🍕 PIZZA CUTTER FOR TEXT 🍕
    
    Imagine you have a looooong piece of paper with writing.
    This function cuts it into smaller pieces at the END of sentences.
    
    Like cutting a pizza! 🍕 We cut at the dots (.), question marks (?), 
    and exclamation marks (!).
    
    Example:
        "I love cats. Dogs are cool!" 
        becomes → ["I love cats.", "Dogs are cool!"]
    """
    # This pattern finds: period, question mark, or exclamation mark
    # followed by a space or end of text
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Remove empty sentences (like empty pizza slices - nobody wants those!)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """
    🔢 COUNT THE WORDS 🔢
    
    Like counting how many toy blocks you have!
    "I love pizza" → 3 blocks (words)
    """
    return len(text.split())


def chunk_text(text: str, document_id: str, url: str, 
               chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    """
    🧩 THE PUZZLE MAKER 🧩
    
    Imagine you have a HUGE story book, too big to carry!
    This function breaks it into smaller puzzle pieces that OVERLAP.
    
    Why overlap? 🤔
    - Imagine cutting a photo of your face in half
    - Your nose might be cut in the middle! 😱
    - Overlap means we include a little bit from the previous piece
    - So nothing important gets lost!
    
    Example (with chunk_size=5 words, overlap=2):
        "The cat sat on the mat and ate fish"
        
        Chunk 1: "The cat sat on the"     (words 1-5)
        Chunk 2: "on the mat and ate"     (words 4-8, overlaps "on the")
        Chunk 3: "and ate fish"           (words 7-9, overlaps "and ate")
    
    Args:
        text: The big story to break up
        document_id: A special name tag for this document
        url: Where this story came from (website address)
        chunk_size: How many words in each piece (default: 400)
        overlap: How many words to repeat between pieces (default: 50)
    
    Returns:
        List of puzzle pieces (chunks)
    """
    # Step 1: Cut the story into sentences (using our pizza cutter!)
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []  # The sentences in our current puzzle piece
    current_word_count = 0        # How many words so far
    
    for sentence in sentences:
        sentence_words = count_words(sentence)
        
        # If adding this sentence makes the chunk too big...
        if current_word_count + sentence_words > chunk_size and current_chunk_sentences:
            # Save the current chunk (it's full!)
            chunk_text_str = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                chunk_index=len(chunks),
                doc_id=document_id,
                url=url,
                token_count=current_word_count
            ))
            
            # Start a new chunk, but keep some sentences for overlap!
            # (Like keeping the edge pieces when starting a new puzzle section)
            overlap_sentences = []
            overlap_count = 0
            
            # Go backwards through sentences to create overlap
            for s in reversed(current_chunk_sentences):
                if overlap_count + count_words(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_count += count_words(s)
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_count
        
        # Add the sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_words
    
    # Don't forget the last chunk! (The last puzzle piece)
    if current_chunk_sentences:
        chunk_text_str = ' '.join(current_chunk_sentences)
        chunks.append(Chunk(
            text=chunk_text_str,
            chunk_index=len(chunks),
            doc_id=document_id,
            url=url,
            token_count=current_word_count
        ))
    
    return chunks


# Old simple function (kept for backwards compatibility)
def tokenizer(text: str) -> List[str]:
    """Simple word splitter - like the old way of cutting pizza with a knife"""
    return text.split()