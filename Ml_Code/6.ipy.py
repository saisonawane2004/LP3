
#Experiment 6.Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset.
# Step 1: Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

%matplotlib inline

# Step 2: Reading the dataset
df = pd.read_csv(r'D:\Ml_Code\ML\ML\Practical B6\sales_data_sample.csv', encoding='Latin-1')
print(df.head())

# Step 3: Creating Datalist
cols = ["PRICEEACH", "SALES"]
data = df[cols]
print(data.head())

# Step 4: Applying K-Means algorithm and finding elbow points
K = range(1, 7)
wss = []

for k in K:
    kmeans = cluster.KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans = kmeans.fit(data)
    wss_iter = kmeans.inertia_   # Within-cluster sum of squares
    wss.append(wss_iter)

# Step 5: Creating Clusters DataFrame
mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
print(mycenters)

# Step 6: Plotting the elbow plot
sns.scatterplot(x='Clusters', y='WSS', data=mycenters, marker='o', s=100, color='blue')
plt.plot(K, wss, linestyle='--', color='red')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.grid(True)
plt.show()
