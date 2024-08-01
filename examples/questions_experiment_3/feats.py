import random


from guided_categorical_embeddings.guided import MultiLabelGuidedEmbeddingsTransformer
from guided_categorical_embeddings.viz import tsne_viz, pca_viz
from ovos_gguf_embeddings import GGUFTextEmbeddingsStore

EMB_MODEL = "paraphrase-multilingual-minilm-l12-v2"
model = GGUFTextEmbeddingsStore(model=EMB_MODEL,
                                skip_db=True,
                                n_gpu_layers=-1)

transformer = MultiLabelGuidedEmbeddingsTransformer.load(f"{EMB_MODEL}_questions_embedding_transformer.pkl")
print(transformer.predict([
    model.get_text_embeddings("qual é a coisa"),
    model.get_text_embeddings("what is the joke"),
    model.get_text_embeddings("conta uma piada"),
    model.get_text_embeddings("conta-me uma anedota por favor"),
    model.get_text_embeddings("podes contar uma piada"),
    model.get_text_embeddings("posso contar uma anedota")
]))

exit()
X = []
y = []
y2 = []

labels = []
with open("utterance_tags_v0.2.csv") as f:
    lines = f.read().split("\n")[1:]
with open("utterance_tags_v0.2_pt.csv") as f:
    lines += f.read().split("\n")[1:]
    random.shuffle(lines)
    for l in lines:
        label, sent = l.split(",", 1)
        labels.append(label)
        try:
            label, sublabel = label.split(":")
        except:
            print(label, sublabel, sent)
            continue
        y.append(label)
        y2.append(sublabel)
        X.append(model.get_text_embeddings(sent))


print(len(X), X[-1])

print("loaded", len(X), "datapoints")

# Create and train the MLP embedding transformer with embedding size specified
transformer = MultiLabelGuidedEmbeddingsTransformer(
    n_embedders=2,
    hidden_layer_sizes_list=[(254, 128), (254, 128)],
    hidden_layer_index_list=[1, 1],
    embedding_size_list=[32, 64],  # size of desired output vector
    max_iter=500)

transformer.fit(X, [y, y2])


# command, question, sentence
main_embeddings = transformer.transform(X, layer=0)
sub_embeddings = transformer.transform(X, layer=1)

# Print the resulting embeddings shape
print(set(y))
print("Embeddings shape:", main_embeddings.shape)
print(set(y2))
print("Embeddings shape:", sub_embeddings.shape)
# Embeddings shape: (500, 64)

# Save the transformer to a file
transformer.save(f"{EMB_MODEL}_questions_embedding_transformer.pkl")


def viz_layer1():
    tsne_viz(main_embeddings, y, f"{EMB_MODEL}_l1_main_tsne_visualization.png")
    pca_viz(main_embeddings, y, f"{EMB_MODEL}_l1_main_pca_visualization.png")

    tsne_viz(main_embeddings, y2, f"{EMB_MODEL}_l1_sub_tsne_visualization.png")
    pca_viz(main_embeddings, y2, f"{EMB_MODEL}_l1_sub_pca_visualization.png")


def viz_layer2():
    tsne_viz(sub_embeddings, y, f"{EMB_MODEL}_l2_main_tsne_visualization.png")
    pca_viz(sub_embeddings, y, f"{EMB_MODEL}_l2_main_pca_visualization.png")

    tsne_viz(sub_embeddings, y2, f"{EMB_MODEL}_l2_sub_tsne_visualization.png")
    pca_viz(sub_embeddings, y2, f"{EMB_MODEL}_l2_sub_pca_visualization.png")


viz_layer1()
viz_layer2()
