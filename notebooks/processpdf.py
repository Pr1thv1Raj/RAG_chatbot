from pypdf import PdfReader
import os
import json
import nltk
from nltk.tokenize import sent_tokenize

file_path = os.path.join("..","data","AI_training_document.pdf")
reader = PdfReader(file_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

# print(text[:500])



nltk.download('punkt_tab')


sentences = sent_tokenize(text)

print(sentences[0])


def create_chunks(sentences, min_words=100, max_words=300):
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) <= max_words:
            current_chunk.append(sentence)
            current_word_count += len(words)
        else:
            if current_word_count >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = len(words)
            else:
                current_chunk.append(sentence)
                current_word_count += len(words)

    # Add any remaining chunk
    if current_chunk and current_word_count >= min_words:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = create_chunks(sentences=sentences)

def save_chunks_to_json(chunks,out_path):
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(chunks,f,indent=2,ensure_ascii=False)

chunk_path = "../chunks/chunks.json"
save_chunks_to_json(chunks,chunk_path)
