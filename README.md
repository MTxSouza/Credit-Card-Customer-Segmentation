<img src="/figures/segmentation.png">
The sample dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.

## Columns
|Column name|description|
|-|-|
|CUST_ID|Identification of Credit Card holder.|
|BALANCE|Balance amount left in their account to make purchases.|
|BALANCE_FREQUENCY|How frequently the Balance is updated.|
|PURCHASES|Amount of purchases made from account.|
|ONEOFF_PURCHASES|Maximum purchase amount done in one-go.|
|INSTALLMENTS_PURCHASES|Amount of purchase done in installment.|
|CASH_ADVANCE|Cash in advance given by the user.|
|PURCHASES_FREQUENCY|How frequently the Purchases are being made.|
|ONEOFF_PURCHASES_FREQUENCY|How frequently Purchases are happening in one-go.|
|PURCHASES_INSTALLMENTS_FREQUENCY|How frequently purchases in installments are being done.|
|CASH_ADVANCE_FREQUENCY|How frequently the cash in advance being paid.|
|CASH_ADVANCE_TRX|Number of Transactions made with "Cash in Advanced".|
|PURCHASES_TRX|Number of purchase transactions made.|
|CREDIT_LIMIT|Limit of Credit Card for user.|
|PAYMENTS|Amount of Payment done by user.|
|MINIMUM_PAYMENTS|Minimum amount of payments made by user.|
|PRC_FULL_PAYMENT|Percent of full payment paid by user.|
|TENURE|Tenure of credit card service for user.|

## Data Preprocessing
Initially, I focused on to handle all missing values on dataset. I noticed that, most of these columns was entirely complete except `CREDIT_LIMIT` with one missing value and `MINIMUM_PAYMENTS` with around 300. The `CREDIT_LIMIT` was easily filled with the mode of the distribution *(most commom value)* and the remaining values from `MINIMUM_PAYMENTS` I decided to remove them just for practicality.

Some variables had completly different scale compared to others, some of them had a distribution between 0-1, 0-300, 0-20000 and 0-40000. Before scaled these variables, I noticed that some variables had a long tail in some directions in KDE *(Kernel Density Estimation)* graph, looks like a log behavior. I plotted some boxplots graphs to visualize the outliers of each variable and I noticed that it had some samples that could harm the normalization step and training stage of cluster algorithm, so I filtered those data and removed them.

Before run any cluster experiment I normalize my dataset using two different approachs.

### Standard Scaler
> It normalizes the distribution centering the mean approximatly to zero and variance to one. It maintain the shape of the distribution and it relasionship among other variables. It is highly influenced by outliers so I removed them as I said before.

### Log Scaler
> It converts a log-normal distribution to a gaussian *(normal)* distribution. It alters the shape of the entire distribution and can alter the relationship among other variables.

After these transformations I decided to use PCA *(Principal Component Analysis)* to reduce the dimensionallity of the dataset, preserving 98.7% of the real variance of the dataset. But before it, I removed some redundant variables from dataset manually using Pearson Correlation among all numerical continues variables *(most of the variables in dataset)*.

The PCA could reduce the dimensionallity of both datasets *(the datasets scaled with std and log scaler)* from 18 variables to 3~5 components. It reduces the complexity of the dataset and reduce the training stage.

The cluster algorithm used to try to solve this problem was `KMeans`. It computes the euclidian distance between each point in dataset and all N centroids *(number of clusters to group the data)* and classify the data to the closest centroid. After this process, the centroids are re-computed *(the mean of the data distribution of the clustered points)*, and then, this process occours again. To evaluate the best model I used the `silhouette score` and `davies_bouldin_score`, both metrics are used when the real label of clusters are unknown.

### Silhouette score
It is defined for each sample:

$$
s={{b - a} \over max(a, b)}
$$

Where:
- $a$: Mean distance of a sample and all other points from the same cluster.
- $b$: Mean distance of a sample and all other points from a different cluster.

> Its value ranges from -1 to 1. Where 1 means that clusters are clearly distinguished, 0 means the distance between clusters is not significant and -1 means wrong clusters.

### Davies Bouldin score
$$
R_{ij}={{s_i + s_j} \over d_{ij}}
$$

$$
DB = {{1 \over k} \sum_{i=1}^k \max_{i \neq j} R_{ij}}
$$

Where:
- $s_i$: Average distance between each point cluster $i$ and the centroid of that cluster.
- $d_{ij}$: Distance between cluster centroids $i$ and $j$.

> It is defined as a ratio between the cluster scatter and the cluster's separation and a lower value will mean that the clustering is better.

<img src="/figures/std_scaler_model.png">
<img src="/figures/log_scaler_model.png">

## Cluster
