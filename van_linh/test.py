from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    r"C:\Users\ADMIN\models\vietnamese-sbert",
    local_files_only=True,
)

vectors = model.encode(
    ["Xin chao", "Day la mot van ban de test"],
    normalize_embeddings=True,
)

print(vectors.shape)