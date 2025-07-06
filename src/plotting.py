import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_product_distribution(df_filtered):
    all_products = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    existing_counts = df_filtered['Product'].value_counts()
    product_counts = pd.Series(index=all_products, data=0)
    for product, count in existing_counts.items():
        if product in product_counts.index:
            product_counts[product] = count
    product_counts = product_counts.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=product_counts.values, y=product_counts.index,
                    palette="viridis", order=product_counts.index)
    plt.title('Complaint Distribution by Product (Sorted)', fontsize=16)
    plt.xlabel('Number of Complaints', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    for i, v in enumerate(product_counts.values):
        ax.text(v + max(product_counts.values)*0.01, i, f"{v:,}",
                color='black', va='center', fontweight='bold')
    plt.tight_layout()
    return plt