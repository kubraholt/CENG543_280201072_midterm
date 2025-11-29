# src/visualize_embeddings.py
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_2d(z, labels, outpath, title):
    plt.figure(figsize=(8,8))
    colors = ['tab:blue','tab:orange']
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(z[idx,0], z[idx,1], s=6, alpha=0.6, label=str(lab), c=colors[int(lab)])
    plt.legend(title='label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_npz', required=True, help='.npz with arrays embeddings,labels')
    parser.add_argument('--out_dir', default='outputs/emb_viz')
    parser.add_argument('--prefix', default='emb')
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--perplexity', type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.emb_npz)
    X = data['embeddings']
    y = data['labels'].astype(int)
    os.makedirs(args.out_dir, exist_ok=True)

    # PCA (2 components)
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X)
    plot_2d(X_pca2, y, os.path.join(args.out_dir, f'{args.prefix}_pca.png'), f'PCA: {args.prefix}')

    # PCA ile boyutu indirip t-SNE
    if args.pca_dim and X.shape[1] > args.pca_dim:
        X_reduced = PCA(n_components=args.pca_dim).fit_transform(X)
    else:
        X_reduced = X
    tsne = TSNE(
    n_components=2,
    perplexity=args.perplexity,
    init='pca',
    learning_rate='auto',
    max_iter=1000
)
    X_tsne = tsne.fit_transform(X_reduced)
    plot_2d(X_tsne, y, os.path.join(args.out_dir, f'{args.prefix}_tsne.png'), f't-SNE: {args.prefix}')

    print("Saved visualizations to", args.out_dir)

if __name__ == '__main__':
    main()
