from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "stjiris/t5-portuguese-legal-summarization"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

text = """
O Tribunal Superior discutiu a aplicação retroativa da nova súmula em casos já julgados.
A decisão tem impacto em milhares de processos pendentes e concluiu que a medida é válida.
"""

input_text = "summarize: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

summary_ids = model.generate(
    inputs,
    num_beams=4,
    min_length=30,
    max_length=100,
    early_stopping=True,
    no_repeat_ngram_size=2
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Resumo abstrativo:", summary)
