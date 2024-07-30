# guiding 3 layer embeddings

dataset
```csv
binary_label,playback_label,media_label,template
not-media,question,not_media,which {author_name} starred in {series_name}
not-media,question,not_media,which {actor_name} invented {silent_movie_name}
not-media,question,not_media,which {actor_name} featured in {series_name}
...
```

## Layer 1

detects if query is media related

### Media vs Not Media

![](l1_binary_pca_visualization.png)
![](l1_binary_tsne_visualization.png)

### Playback Types

![](l1_playback_pca_visualization.png)
![](l1_playback_tsne_visualization.png)

### Media  Types

![](l1_media_pca_visualization.png)
![](l1_media_tsne_visualization.png)


## Layer 2

distinguishes GAME, VIDEO, AUDIO, IOT_DEVICE

### Media vs Not Media

![](l2_binary_pca_visualization.png)
![](l2_binary_tsne_visualization.png)

### Playback Types

![](l2_playback_pca_visualization.png)
![](l2_playback_tsne_visualization.png)
![](l2_playback_3d_tsne_visualization.png)

### Media  Types

![](l2_media_pca_visualization.png)
![](l2_media_tsne_visualization.png)



## Layer 3

distinguishes MOVIE, MUSIC, PODCAST ....

### Media vs Not Media

![](l3_binary_pca_visualization.png)
![](l3_binary_tsne_visualization.png)

### Playback Types

![](l3_playback_pca_visualization.png)
![](l3_playback_tsne_visualization.png)
![](l3_playback_3d_tsne_visualization.png)

### Media  Types

![](l3_media_pca_visualization.png)
![](l3_media_tsne_visualization.png)


