from transformers import T5Tokenizer, T5ForConditionalGeneration

# Tokenizador e modelo treinados em português para summarização
tokenizer = T5Tokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
model = T5ForConditionalGeneration.from_pretrained("recogna-nlp/ptt5-base-summ")

texto = """
O Brasil é uma das maiores economias da América Latina, com forte presença no agronegócio, mineração 
e setor de serviços. Recentemente, o Banco Central sinalizou uma tendência de queda na taxa de juros, 
como parte da estratégia de controle da inflação. A estabilidade política também tem sido vista como 
um fator relevante para atrair investimentos estrangeiros, contribuindo para o cenário econômico do país.
"""

inputs = tokenizer.encode("summarize: " + texto, max_length=512, truncation=True, return_tensors="pt")
summary_ids = model.generate(
    inputs,
    max_length=150,
    min_length=30,
    num_beams=4,
    no_repeat_ngram_size=3,
    early_stopping=True
)

resumo = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(resumo)
