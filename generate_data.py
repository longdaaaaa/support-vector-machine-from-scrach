import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def generate_data(n_samples=100, centers=2, cluster_std=1.5, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, y

if __name__ == "__main__":
    X, y = generate_data()

    # VISUALIZE
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.title('Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('generated_data.png')
    plt.show()

    # SAVE AS CSV
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Label'] = y
    df.to_csv('svm_data.csv', index=False)

    print("Data generated and saved to svm_data.csv")
