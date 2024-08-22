import pandas as pd
import matplotlib.pyplot as plt

def categorize_dockq_score(score):
    """Categorize the DockQ score into quality bins."""
    if 0.00 <= score < 0.23:
        return "Incorrect"
    elif 0.23 <= score < 0.49:
        return "Acceptable quality"
    elif 0.49 <= score < 0.80:
        return "Medium quality"
    elif score >= 0.80:
        return "High quality"
    else:
        return "Unknown"

def plot_dockq_distribution(csv_file):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file)
    
    # Ensure the dockq column is float type
    df['dockq'] = df['dockq'].astype(float)
    
    # Categorize DockQ scores
    df['category'] = df['dockq'].apply(categorize_dockq_score)
    
    # Count the number of structures in each category
    category_counts = df['category'].value_counts()
    
    # Plot the distribution
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color=['red', 'orange', 'yellow', 'green'], edgecolor='black')
    plt.title('Distribution of DockQ Scores')
    plt.xlabel('DockQ Quality')
    plt.ylabel('Number of Structures')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
csv_file = 'dockq_scores.csv'  # Path to your CSV file
plot_dockq_distribution(csv_file)
