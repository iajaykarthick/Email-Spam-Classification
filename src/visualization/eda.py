import matplotlib.pyplot as plt


def plot_class_distribution(df, target):
    # Calculate the value counts and their proportions
    counts = df[target].value_counts()
    proportions = counts / len(df)

    # Create the bar plot
    ax = proportions.plot(kind='bar')
    plt.ylabel('Proportion')
    plt.xlabel('Class')
    plt.title('Class Distribution')

    # Annotate the bars with the percentage
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center')

    plt.show()
