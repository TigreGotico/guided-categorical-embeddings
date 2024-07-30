import json
import random

from guided_categorical_embeddings.guided import LabelGuidedEmbeddingsTransformer
from guided_categorical_embeddings.viz import tsne_viz, umap_viz, pca_viz, scatter_viz, tsne_3d_viz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

for dataset in ["ocp_features.json",
                "playback_ocp_features.json",
                "binary_ocp_features.json"]:
    print("\n####", dataset)
    # randomly drop N keys (keeping final label) from dict features for data augmentation
    CORRUPT = 1

    # json files contains dicts of pre-extracted categorical features
    with open(dataset) as f:
        DATASET = json.load(f)

    X = []
    y = []
    X_aug = []
    y_aug = []
    for label, samples in DATASET.items():
        for s in samples:
            assert isinstance(s, dict)
            y.append(label)
            X.append(dict(s))
            X_aug.append(dict(s))
            y_aug.append(label)
            if CORRUPT:
                for i in range(CORRUPT):
                    keys = list(s.keys())
                    if len(keys) < 2:
                        break
                    s.pop(random.choice(keys))
                    X_aug.append(dict(s))
                    y_aug.append(label)

    print("loaded", len(X), "datapoints")
    # Create and train the MLP embedding transformer with embedding size specified
    transformer = LabelGuidedEmbeddingsTransformer(hidden_layer_sizes=(256, 128, 64),
                                                   hidden_layer_index=2,
                                                   embedding_size=3 * len(set(y)),  # size of desired output vector
                                                   max_iter=1000)
    transformer.fit(X_aug, y_aug)
    embeddings = transformer.transform(X)

    # Print the resulting embeddings shape
    print("Embeddings shape:", embeddings.shape)
    # Embeddings shape: (500, 64)

    # Save the transformer to a file
    transformer.save(f'{dataset.split(".json")[0]}_embedding_transformer.pkl')

    ## Visualization of embeddings
    tsne_3d_viz(embeddings, y, f"{dataset.split('.json')[0]}_3d_tsne_visualization.png")
    tsne_viz(embeddings, y, f"{dataset.split('.json')[0]}_tsne_visualization.png")
    umap_viz(embeddings, y, f"{dataset.split('.json')[0]}_umap_visualization.png")
    pca_viz(embeddings, y, f"{dataset.split('.json')[0]}_pca_visualization.png")
    scatter_viz(embeddings, y, f"{dataset.split('.json')[0]}_scatteplot_visualization.png")

    # Split the data for training and testing
    X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(transformer.transform(X_aug), y_aug,
                                                                        test_size=0.4)

    # Vectorize the original data
    vectorizer = DictVectorizer(sparse=False)
    X_dict = vectorizer.fit_transform(X_aug)
    X_train_dict, X_test_dict, y_train_dict, y_test_dict = train_test_split(X_dict, y_aug,
                                                                            test_size=0.4)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Perceptron': Perceptron(),
        'MLP Classifier': MLPClassifier(max_iter=200),
        'SVM': SVC(max_iter=200),
        'Random Forest': RandomForestClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    # Evaluate classifiers using dict feats
    print("\n## Evaluating classifiers using dict feats:")
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_dict, y_train_dict)
        y_pred = clf.predict(X_test_dict)
        accuracy = accuracy_score(y_test_dict, y_pred)
        results[name] = accuracy
        print(f"- {name} accuracy: {accuracy:.2f}")

    # Evaluate classifiers using embeddings
    print("\n## Evaluating classifiers using embeddings:")
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_emb, y_train_emb)
        y_pred = clf.predict(X_test_emb)
        accuracy = accuracy_score(y_test_emb, y_pred)
        results[name] = accuracy
        print(f"- {name} accuracy: {accuracy:.2f}")
