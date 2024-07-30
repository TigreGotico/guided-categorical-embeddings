
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def tsne_viz(embeddings, y, path):
    ## Visualization of embeddings
    import matplotlib.pyplot as plt

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_numerical = label_encoder.fit_transform(y)
    unique_labels = label_encoder.classes_

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_numerical, cmap='tab20')
    plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Labels')
    plt.title('t-SNE Visualization of OCP Embeddings')


    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def pca_viz(embeddings, y, path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_numerical = label_encoder.fit_transform(y)

    # Assume `embeddings` is your high-dimensional data
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_numerical,
                cmap='tab20')  # Use a colormap suited for your labels
    plt.colorbar()
    plt.title('PCA Visualization of Embeddings')
    # Save the plot
    plt.savefig(path)
    plt.close()


def umap_viz(embeddings, y, path):
    import matplotlib.pyplot as plt
    import umap

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_numerical = label_encoder.fit_transform(y)

    # Assume `embeddings` is your high-dimensional data
    umap_model = umap.UMAP(n_components=2)
    embeddings_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_numerical,
                cmap='tab20')  # Use a colormap suited for your labels
    plt.colorbar()
    plt.title('UMAP Visualization of Embeddings')
    # Save the plot
    plt.savefig(path)
    plt.close()


def scatter_viz(embeddings, y, path):
    import matplotlib.pyplot as plt

    # Assume `embeddings_2d` is your 2D transformed embeddings and `labels` is the corresponding label
    plt.figure(figsize=(12, 10))
    for label in set(y):
        indices = [i for i, l in enumerate(y) if l == label]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=label)

    plt.legend()
    plt.title('Scatter Plot of Embeddings with Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # Save the plot
    plt.savefig(path)
    plt.close()


def tsne_3d_viz(embeddings, y, path):
    ## Visualization of embeddings
    import matplotlib.pyplot as plt

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_numerical = label_encoder.fit_transform(y)
    unique_labels = label_encoder.classes_

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.manifold import TSNE

    # Assuming `embeddings` is your high-dimensional data and `labels` are the corresponding labels
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    embeddings_tsne_3d = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings_tsne_3d[:, 0], embeddings_tsne_3d[:, 1], embeddings_tsne_3d[:, 2],
                         c=y_numerical, cmap='tab20')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title('3D t-SNE Visualization of Embeddings')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    # Save the plot
    plt.savefig(path)
    plt.close()
