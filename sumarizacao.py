from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Configuração de dispositivo (usa GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregamento do modelo e tokenizer
model_name = "stjiris/t5-portuguese-legal-summarization"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Texto com ~410 palavras
text = """
A Amazônia é a maior floresta tropical do mundo, cobrindo cerca de 5,5 milhões de quilômetros quadrados. Localizada na América do Sul, abrange nove países, sendo a maior parte situada no Brasil. A floresta é conhecida por sua biodiversidade única, abrigando milhões de espécies de plantas, animais e insetos, muitas das quais ainda não foram catalogadas pela ciência. Além de seu valor ecológico, a Amazônia desempenha um papel fundamental na regulação do clima global, atuando como um enorme sumidouro de carbono que ajuda a reduzir os efeitos das mudanças climáticas.

No entanto, nas últimas décadas, a Amazônia tem enfrentado sérias ameaças devido ao desmatamento, à exploração ilegal de madeira, à mineração e à expansão da agropecuária. Dados recentes mostram que o ritmo de desmatamento aumentou significativamente, colocando em risco não apenas o ecossistema local, mas também o equilíbrio ambiental do planeta. O desmatamento contribui para o aumento da emissão de gases de efeito estufa, além de afetar diretamente as comunidades indígenas e tradicionais que dependem da floresta para sua sobrevivência.

Organizações ambientais e governos de diferentes países têm se mobilizado para proteger a floresta. Iniciativas como o monitoramento por satélite, a criação de áreas de conservação e o incentivo a práticas sustentáveis de uso da terra são algumas das estratégias adotadas. No entanto, especialistas afirmam que os esforços ainda são insuficientes diante da magnitude dos desafios enfrentados.

Uma das soluções propostas é o investimento em bioeconomia, que visa utilizar os recursos da floresta de forma sustentável, gerando renda para as populações locais sem causar degradação ambiental. A valorização do conhecimento tradicional dos povos indígenas também é considerada essencial para a conservação da Amazônia, uma vez que essas comunidades têm uma relação ancestral com o meio ambiente e sabem como manejá-lo sem destruí-lo.

Proteger a Amazônia não é apenas uma responsabilidade dos países que a abrigam, mas um dever coletivo da humanidade. O futuro da floresta e de toda a vida que dela depende está nas mãos das gerações atuais, que precisam agir com urgência e responsabilidade para garantir sua preservação.
"""

# Contagem de palavras
word_count = len(text.split())
if word_count < 300 or word_count > 500:
    print(f"❌ O texto deve conter entre 300 e 500 palavras. Atualmente contém {word_count}.")
    exit()

# Calcula limite máximo de palavras no resumo (50%)
max_summary_words = word_count // 2
# Estimativa de tokens: 1 palavra ≈ 1.3 tokens
max_summary_tokens = int(max_summary_words * 1.3)

# Prepara entrada com prefixo "summarize:"
input_text = "summarize: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

# Geração do resumo
summary_ids = model.generate(
    inputs,
    num_beams=4,
    max_length=max_summary_tokens,
    min_length=30,
    early_stopping=True,
    no_repeat_ngram_size=2
)

# Decodificação
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
summary_word_count = len(summary.split())

# Resultados
print("\n📄 Texto original:")
print(f"{word_count} palavras")

print("\n📝 Resumo gerado:")
print(summary)
print(f"\n✅ Resumo contém {summary_word_count} palavras ({(summary_word_count/word_count)*100:.1f}% do original)")
