from typing import Dict, List

from guided_categorical_embeddings.guided import MultiLabelGuidedEmbeddingsTransformer
from guided_categorical_embeddings.viz import tsne_viz, pca_viz
from nltk import word_tokenize, pos_tag


def sentence2dict(sentence: str) -> Dict[str, bool]:
    # Define indicators for different types of sentences
    YES_NO_STARTERS = ['were', 'are', 'will', 'do', 'if', 'have', 'did', 'has', 'does', 'can', 'shall', 'am',
                       'should', 'would', 'was', 'is', 'might', 'must', 'could', 'may', 'would', 'should', 'might']
    QUESTION_STARTERS = ['where', 'when', 'how', 'why', 'who', 'whose', 'which', 'what',
                         'whom', 'how many', 'how much', 'how long', 'how often', 'how far', 'how tall', 'how old',
                         'how big']
    COMMAND_STARTERS = ["don't forget ", "do not forget ", 'please ', 'make sure to ', 'remember to ', 'ensure you ',
                        'do not forget to ', 'don’t forget to ', 'make sure you ']
    DENIAL_STARTERS = ["don't ", "do not ", "no ", 'never ', 'not at all ', 'no way ', 'under no circumstances ']
    THANKS_WORDS = ["thanks", "thank you", 'thanks a lot', 'thanks very much', 'many thanks', 'thanks so much']
    PLEASE_WORDS = ["please", 'kindly', 'would you mind', 'if you could', 'could you please']
    DENIAL_WORDS = ["forbidden", "prohibited", "not permitted", "not allowed", 'unacceptable',
                    'restricted', 'banned', 'out of bounds']
    sentence = sentence.lower().strip()
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    def contains_conjunction(tokens: List[str]) -> bool:
        """Check if the sentence contains conjunctions."""
        conjunctions = {'and', 'or', 'but', 'so', 'for', 'nor', 'yet'}
        return any(token in conjunctions for token in tokens)

    def lexical_diversity(tokens: List[str]) -> float:
        """Calculate lexical diversity of the tokens."""
        return len(set(tokens)) / len(tokens) if tokens else 0

    def pos_tag_counts(tags: List[str]) -> Dict[str, int]:
        """Count occurrences of specific POS tags."""
        pos_counts = {}
        for tag in tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        return pos_counts

    def pos_tag_transitions(tags: List[str]) -> Dict[str, int]:
        """Count transitions between POS tags."""
        transitions = {}
        for i in range(len(tags) - 1):
            pair = (tags[i], tags[i + 1])
            transitions[pair] = transitions.get(pair, 0) + 1
        return transitions

    # Initialize feature dictionary
    feats = {}

    # Extract features based on POS tags and sentence content
    if tokens:
        first_word = tokens[0]
        pos_tags_list = [tag for _, tag in pos_tags]

        feats['contains_conjunction'] = contains_conjunction(tokens)
        feats['lexical_diversity'] = lexical_diversity(tokens)
        feats["is_play"] = first_word in {"play", "watch", "listen", "view", "launch", "open", "start", "activate"}
        feats["is_the"] = first_word == "the"
        feats["is_cmd"] = any(sentence.startswith(prefix) for prefix in COMMAND_STARTERS)
        feats["is_denial"] = any(sentence.startswith(prefix) for prefix in DENIAL_STARTERS)
        feats["thanks_words"] = any(word in sentence for word in THANKS_WORDS)
        feats["denial_words"] = any(word in sentence for word in DENIAL_WORDS)
        feats["please_words"] = any(word in sentence for word in PLEASE_WORDS)
        feats["is_yes_no"] = first_word in YES_NO_STARTERS
        feats['is_wh'] = first_word in QUESTION_STARTERS
        feats['startswith_pron'] = first_word in {"he", "she", "it", "they", "their", "we", "i"}

        # Sentence length and token-based features
        feats['sentence_length'] = len(tokens)
        feats['avg_token_length'] = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        feats['unique_token_count'] = len(set(tokens))

        # POS tag-based features
        pos_counts = pos_tag_counts(pos_tags_list)
        feats.update({
            'num_NN': pos_counts.get('NN', 0),
            'num_NNP': pos_counts.get('NNP', 0),
            'num_NNS': pos_counts.get('NNS', 0),
            'num_NNPS': pos_counts.get('NNPS', 0),
            'num_PRP': pos_counts.get('PRP', 0),
            'num_VB': pos_counts.get('VB', 0),
            'num_VBP': pos_counts.get('VBP', 0),
            'num_VBZ': pos_counts.get('VBZ', 0),
            'num_VBD': pos_counts.get('VBD', 0),
            'num_VBN': pos_counts.get('VBN', 0),
            'num_VBG': pos_counts.get('VBG', 0),
            'num_JJ': pos_counts.get('JJ', 0),
            'num_JJr': pos_counts.get('JJR', 0),
            'num_JJs': pos_counts.get('JJS', 0),
        })

        # POS tag transitions
        transitions = pos_tag_transitions(pos_tags_list)
        for (tag1, tag2), count in transitions.items():
            feats[f'trans_{tag1}_{tag2}'] = count

        # Check for statements
        subject_tags = {"NN", "NNP", "NNS", "NNPS", "PRP"}
        verb_tags = {"VB", "VBP", "VBZ", "VBD", "VBN", "VBG"}
        feats['is_statement'] = any(tag in subject_tags for _, tag in pos_tags[:2]) and \
                                any(tag in verb_tags for _, tag in pos_tags)

        # Check for exclamations and commands
        feats['is_exclamation'] = (first_word == "what" and len(tokens) > 1 and tokens[1] in {"a", "an"}) or \
                                  (first_word == "how" and any(tag == "JJ" for _, tag in pos_tags[:2]))

        feats['is_command'] = any(tag in verb_tags for _, tag in pos_tags[:1]) and \
                              not any(tag in subject_tags for _, tag in pos_tags[:2])

        # Check for polite requests
        feats['is_request'] = first_word in {"would", "could", "can"} and any(tag == "PRP" for _, tag in pos_tags[1:2])

        # Social expressions
        social_expressions_starters = {"have", "get", "help"}
        feats['is_social_expression'] = first_word in social_expressions_starters and \
                                        any(tag == "VB" for _, tag in pos_tags[:1])

    return feats


X = []
y = []
y2 = []

labels = []
with open("utterance_tags_v0.2.csv") as f:
    lines = f.read().split("\n")[1:]
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
        X.append(sentence2dict(sent))

print(len(X), X[-1])

print("loaded", len(X), "datapoints")
# Create and train the MLP embedding transformer with embedding size specified
transformer = MultiLabelGuidedEmbeddingsTransformer(
    n_embedders=2,
    hidden_layer_sizes_list=[(16,), (32,)],
    hidden_layer_index_list=[0, 0],
    embedding_size_list=[8, 8],  # size of desired output vector
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
transformer.save('questions_embedding_transformer.pkl')


def viz_layer1():
    tsne_viz(main_embeddings, y, "l1_main_tsne_visualization.png")
    pca_viz(main_embeddings, y, "l1_main_pca_visualization.png")

    tsne_viz(main_embeddings, y2, "l1_sub_tsne_visualization.png")
    pca_viz(main_embeddings, y2, "l1_sub_pca_visualization.png")


def viz_layer2():
    tsne_viz(sub_embeddings, y, "l2_main_tsne_visualization.png")
    pca_viz(sub_embeddings, y, "l2_main_pca_visualization.png")

    tsne_viz(sub_embeddings, y2, "l2_sub_tsne_visualization.png")
    pca_viz(sub_embeddings, y2, "l2_sub_pca_visualization.png")


viz_layer1()
viz_layer2()
