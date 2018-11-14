from sklearn import manifold
import matplotlib.pyplot as plt

def t_SNE_visualization(X, n_gram_dict=None, target_dimension=2):
    '''
    use t-SNE to reduce dimension of X and visualizing embedding reuslt
    :param X:
    :return:
    '''

    tsne = manifold.TSNE(n_components=target_dimension, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Original data dimension is {}.\
           Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # normalization for easier display
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))

    val = len(n_gram_dict.keys())
    if n_gram_dict is not None:
        for key in n_gram_dict:
            plt.text(X_norm[n_gram_dict[key], 0], X_norm[n_gram_dict[key], 1],
                     str(key), color=plt.cm.Set1(n_gram_dict[key]/val),
                     fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.show()