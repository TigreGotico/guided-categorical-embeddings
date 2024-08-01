import random
from guided_categorical_embeddings.guided import LabelGuidedEmbeddingsTransformer, MultiLabelGuidedEmbeddingsTransformer
from guided_categorical_embeddings.viz import tsne_viz, umap_viz, pca_viz, scatter_viz, tsne_3d_viz


dataset = "ocp_media_templates_en.csv"
print("\n####", dataset)

with open(dataset) as f:
    lines = f.read().split("\n")[1:]

    # randomly drop N keys (keeping final label) from dict features for data augmentation
    CORRUPT = 1

    X = []
    y = []
    y2 = []
    y3 = []
    X_aug = []
    y_aug = []
    y2_aug = []
    y3_aug = []
    for l in lines:
        binary_label, playback_label, media_label, sent = l.split(",")
        toks = sent.split()
        s = {f"contains_{k}": True for k in sent.split()}

        X.append(dict(s))

        y.append(binary_label)
        y2.append(playback_label)
        y3.append(media_label)

        X_aug.append(dict(s))
        y_aug.append(binary_label)
        y2_aug.append(playback_label)
        y3_aug.append(media_label)
        if CORRUPT:
            for i in range(CORRUPT):
                keys = list(s.keys())
                if len(keys) < 2:
                    break
                s.pop(random.choice(keys))
                X_aug.append(dict(s))
                y_aug.append(binary_label)
                y2_aug.append(playback_label)
                y3_aug.append(media_label)

    print("loaded", len(X), "datapoints")
    # Create and train the MLP embedding transformer with embedding size specified
    transformer = MultiLabelGuidedEmbeddingsTransformer(
        n_embedders=3,
        hidden_layer_sizes_list=[(64, 32), (128, 64), (256, 128, 64)],
        hidden_layer_index_list=[1, 1, 2],
        embedding_size_list=[32, 64, 128],  # size of desired output vector
        max_iter=1000)

    transformer.fit(X_aug, [y_aug, y2_aug, y3_aug])
    media_embeddings = transformer.transform(X, layer=2)
    playback_embeddings = transformer.transform(X, layer=1)
    bin_embeddings = transformer.transform(X, layer=0)

    # Print the resulting embeddings shape
    print("Embeddings shape:", media_embeddings.shape)
    # Embeddings shape: (500, 64)

    # Save the transformer to a file
    transformer.save(f'{dataset.split(".json")[0]}_embedding_transformer.pkl')

    def viz_layer3():
        ## Visualization of embeddings
        tsne_3d_viz(media_embeddings, y3, "l3_media_3d_tsne_visualization.png")
        tsne_viz(media_embeddings, y3, "l3_media_tsne_visualization.png")
        umap_viz(media_embeddings, y3, "l3_media_umap_visualization.png")
        pca_viz(media_embeddings, y3, "l3_media_pca_visualization.png")

        tsne_3d_viz(media_embeddings, y2, "l3_playback_3d_tsne_visualization.png")
        tsne_viz(media_embeddings, y2, "l3_playback_tsne_visualization.png")
        umap_viz(media_embeddings, y2, "l3_playback_umap_visualization.png")
        pca_viz(media_embeddings, y2, "l3_playback_pca_visualization.png")

        tsne_3d_viz(media_embeddings, y, "l3_binary_3d_tsne_visualization.png")
        tsne_viz(media_embeddings, y, "l3_binary_tsne_visualization.png")
        umap_viz(media_embeddings, y, "l3_binary_umap_visualization.png")
        pca_viz(media_embeddings, y, "l3_binary_pca_visualization.png")


    def viz_layer2():
        ## Visualization of embeddings
        tsne_3d_viz(playback_embeddings, y3, "l2_media_3d_tsne_visualization.png")
        tsne_viz(playback_embeddings, y3, "l2_media_tsne_visualization.png")
        umap_viz(playback_embeddings, y3, "l2_media_umap_visualization.png")
        pca_viz(playback_embeddings, y3, "l2_media_pca_visualization.png")

        tsne_3d_viz(playback_embeddings, y2, "l2_playback_3d_tsne_visualization.png")
        tsne_viz(playback_embeddings, y2, "l2_playback_tsne_visualization.png")
        umap_viz(playback_embeddings, y2, "l2_playback_umap_visualization.png")
        pca_viz(playback_embeddings, y2, "l2_playback_pca_visualization.png")

        tsne_3d_viz(playback_embeddings, y, "l2_binary_3d_tsne_visualization.png")
        tsne_viz(playback_embeddings, y, "l2_binary_tsne_visualization.png")
        umap_viz(playback_embeddings, y, "l2_binary_umap_visualization.png")
        pca_viz(playback_embeddings, y, "l2_binary_pca_visualization.png")

    def viz_layer1():
        ## Visualization of embeddings
        tsne_3d_viz(bin_embeddings, y3, "l1_media_3d_tsne_visualization.png")
        tsne_viz(bin_embeddings, y3, "l1_media_tsne_visualization.png")
        umap_viz(bin_embeddings, y3, "l1_media_umap_visualization.png")
        pca_viz(bin_embeddings, y3, "l1_media_pca_visualization.png")
        scatter_viz(bin_embeddings, y3, "l1_media_scatterplot_visualization.png")

        tsne_3d_viz(bin_embeddings, y2, "l1_playback_3d_tsne_visualization.png")
        tsne_viz(bin_embeddings, y2, "l1_playback_tsne_visualization.png")
        umap_viz(bin_embeddings, y2, "l1_playback_umap_visualization.png")
        pca_viz(bin_embeddings, y2, "l1_playback_pca_visualization.png")
        scatter_viz(bin_embeddings, y2, "l1_playback_scatterplot_visualization.png")

        tsne_3d_viz(bin_embeddings, y, "l1_binary_3d_tsne_visualization.png")
        tsne_viz(bin_embeddings, y, "l1_binary_tsne_visualization.png")
        umap_viz(bin_embeddings, y, "l1_binary_umap_visualization.png")
        pca_viz(bin_embeddings, y, "l1_binary_pca_visualization.png")
        scatter_viz(bin_embeddings, y, "l1_binary_scatterplot_visualization.png")

    viz_layer3()
    viz_layer2()
    viz_layer1()