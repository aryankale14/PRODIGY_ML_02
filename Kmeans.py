import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Load the data
df=pd.read_csv("Mall_Customers.csv")

#Print head and tail of dataset
print(df.head())
print(df.tail())

#Checking data types of dataset
print(df.shape)
print(df.info())
print(df.describe())

#Check for missing values
df.isnull().sum()


#Plotting all the columns 

#Age Distribution
fig1=px.histogram(df,x='Age', nbins=10,title='Distribution of Age')
fig1.show()

#Annual Income distribution
fig2 = px.histogram(df, x='Annual Income (k$)', nbins=10, title='Distribution of Annual Income')
fig2.show()

#Spending Score distribution
fig3 = px.histogram(df, x='Spending Score (1-100)', nbins=10, title='Distribution of Spending Score')
fig3.show()

#Age vs Annual Income
fig4 = px.scatter(df, x='Age', y='Annual Income (k$)', color='Gender', hover_data=['CustomerID'])
fig4.update_layout(title='Age vs Annual Income')
fig4.show()

#Annual Income vs Spending Score
fig5 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Gender', hover_data=['CustomerID'])
fig5.update_layout(title='Annual Income vs Spending Score')
fig5.show()

#Convert Gender into categorical numerical values
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

#Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


#Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#Plot the Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying K-means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

#Add the cluster labels to the original dataframe
df['Cluster'] = y_kmeans

#Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


