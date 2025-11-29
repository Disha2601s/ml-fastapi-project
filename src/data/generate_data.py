import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

os.makedirs('data', exist_ok=True)

# Create synthetic classification data with one categorical column 'education' for slice metrics
X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, random_state=42)

df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
df['target'] = y

# Create a synthetic categorical column with 4 categories
np.random.seed(42)
df['education'] = np.random.choice(['Primary', 'Secondary', 'Bachelors', 'Masters'], size=len(df), p=[0.2,0.3,0.3,0.2])

# Save to CSV
df.to_csv('data/dataset.csv', index=False)
print('Wrote data/dataset.csv with', len(df), 'rows')