#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Load results
DF = pd.read_csv('output/lightweight/results.csv')

# Define styles
colors = {2.0: 'blue', 16.0: 'green'}
markers = {0.01: 'o', 0.05: 's'}

plt.figure(figsize=(8,6))
for _, row in DF.iterrows():
    plt.scatter(
        row['compression_cost'],
        row['val_loss'],
        color=colors[row['budget']],
        marker=markers[row['entropy_coef']],
        s=100,
        label=f"E={row['entropy_coef']},B={row['budget']}"
    )

# Build unique legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title='Entropy,Budget', bbox_to_anchor=(1.05,1), loc='upper left')

plt.xlabel('Compression Cost')
plt.ylabel('Validation Loss')
plt.title('Figure 1: Performance vs Compression Trade-off')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/lightweight/figure1_tradeoff.png', dpi=300)
plt.show()
