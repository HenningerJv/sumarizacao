from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Seleciona GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelo e tokenizer
model_name = "stjiris/t5-portuguese-legal-summarization"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Texto a ser resumido
text = """
O Tribunal Superior discutiu a aplicação retroativa da nova súmula em casos já julgados.
A decisão tem impacto em milhares de processos pendentes e concluiu que a medida é válida.
"""

# Prepara entrada com prefixo "summarize:"
input_text = "summarize: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

# Geração do resumo
summary_ids = model.generate(
    inputs,
    num_beams=4,
    min_length=20,
    max_length=80,
    early_stopping=True,
    no_repeat_ngram_size=2
)

# Decodifica o resumo
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Resumo abstrativo:", summary)
