from typing import List 
from Models.models import Chunk , overlappingChunk
import re
def simple_chunking(state : Chunk) -> List[str]:
    '''
    - Fixed Chunking
    '''
    chunked_text = state.text.strip()

    if not chunked_text:
        return []
    chunks : List[str] = []

    for i in range(0, len(chunked_text), state.chunk_size):
        chunk = chunked_text[i:i+state.chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks


def overlap_chunking(state : overlappingChunk) -> List[str]:
    '''
    - Fixed-size chunking with overlap (by characters).
    '''
    text = state.text
    if not text:
        return []
    
    if state.overlapping_size > state.chunk_size:
        raise Exception("Not Possible to do this ")
    step_size = state.chunk_size - state.overlapping_size

    chunks: List[str] = []
    for i in range(0, len(text), step_size):
        chunk = text[i:i + state.chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def sentence_aware_chunking(state : Chunk) ->List[str]:
    '''
    - Splits into sentences 
    - Packs sentences into chunks up to chunk_size 
    - Preserves meaning boundaries
    '''
    chunked_text = state.text.strip()
    if not chunked_text:
        return []

    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
    sentences = SENT_SPLIT.split(chunked_text)
    chunks = []
    current_chunk = ""


    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue


        if (len(sentence) > state.chunk_size):
            if current_chunk :
                current_chunk = current_chunk.strip()
                chunks.append(current_chunk)
                current_chunk = ""

            for i in range(0 , len(sentence) , state.chunk_size):
                inner_chunk = sentence[i:i + state.chunk_size]

                if inner_chunk:
                    chunks.append(inner_chunk)
            continue
        if len(current_chunk) + len(sentence) + 1 <= state.chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
