from colpali_engine.models import ColDeepSeekOCR, ColDeepSeekOCRProcessor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
model = ColDeepSeekOCR.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
processor = ColDeepSeekOCRProcessor(tokenizer)

pil_images = ['sample_image.png']
# Embed document images
doc_inputs = processor.process_images(pil_images)
doc_embeddings = model(**doc_inputs)  # (batch, seq, 128)

# Embed queries
query_inputs = processor.process_texts(["What is the batch norm paper?"])
query_embeddings = model(**query_inputs)  # (batch, seq, 128)

scores = processor.score(query_embeddings, doc_embeddings)