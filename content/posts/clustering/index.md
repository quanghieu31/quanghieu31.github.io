---
title: "Flat, Hierarchical, and Model-based Clustering"
date: 2025-03-29T12:00:00Z
draft: false
layout: "single"
mathjax: true
---

[My notes from CMSC 35400: Machine Learning, with Professor Risi Kondor, at the University of Chicago.]

I realized I have used K-means clustering for too many times but never actually digging into the math of it. This is a chance for me to solidify my understanding of these powerful unsupervised learning methods.

### Clustering: the Good
- It is a natural thing to want to do with large data.
- Can reveal a lot about the structure of data → exploratory data analysis.  
  _e.g., finding new types of stars, patients with similar disease profiles, ..._
- Allows us to compress data by replacing points with their cluster representatives (called **vector quantization**).
- Key part of finding structure in large graphs & networks.
### Clustering: the Bad
- It’s an unsupervised problem → always harder to formalize.
- Ill-defined: different objective functions are possible, no clear winner.  
  Even after clustering, it's hard to say whether the result is good or bad → subjective.
- What is the “correct” number of clusters? Also subjective.  
  Often the data is ambiguous in this regard.
- End users may attribute too much significance to clusters, with unforeseeable consequences.
- Compared to supervised ML, the theory is in its infancy.
### Types of clustering to be discussed:
1. [Flat clustering: K-means](#Flat-clustering-Kmeans)
2. [Hierarchical clustering](#Hierarchical-clustering)
3. [Model-based clustering: Gaussian Mixture Model](#Modelbased-clusteringGaussian-Mixture-Model)
### Flat clustering: $K$-means
- Input: samples ($\vec{x}_1, \vec{x}_2, \cdots, \vec{x}_n$) $\in \mathbb{R}^p$ and choosen $k$ for number of clusters wanted
- Output: $k$ disjoint sets $C_1, C_2, \cdots, C_k$ whose union is $\{\vec{x}_1, \cdots, \vec{x}_n\}$ 
- Clustering depends on a distance metric, i.e. the simplest would be the Euclidean distance or $d(\vec{x}_i , \mu_k) = ||\vec{x}_i - \mu_k||^2 = \sum_{j=1}^p (\vec{x}_{ij} - \mu_{kj})^2$

Problem: Find $C_1,C_2,\ldots,C_k$ and centroids $m_1,m_2,\ldots,m_k \in \mathbb{R}^d$ that minimize this loss function:
$$J_{\text{avg}}^2 = \sum_{j=1}^{k} \sum_{x \in C_j} d(x,m_j)^2$$
where $d(x,m_j) = \|x - m_j\|$.

This is an optimization problem.
- Is it continuous? No.  Is it combinatorial? No.  → It's a mixed problem.
- Is it convex? No.  
- How do we solve it? Use an alternating minimization strategy.

Lloyd’s algorithm ($k$–means):

- ${\vec{m}_1, \vec{m}_2, \cdots, \vec{m}_k}$ as picked from $k$ random samples in the sample space, each is a vector of $p$ dimension (no.features)
- initialize empty sets: $C_1, C_2, \cdots, C_k$
- until convergence (no change in cluster assignments)
	- for $i=1$ to $n$ (assign each sample to the closest center/mean):
		- $j \leftarrow \arg\min_{j\in\{1,\cdots,k\}} d(\vec{x}_i, \vec{m}_j)$  (for center in centers (the $\vec{m}$'s), find the cluster/centers that gives the smallest distance from the current considered sample $\vec{x}_i$)
		- $C_j \leftarrow C_j \cup \{\vec{x}_i\}$ (add the sample point to the cluster set)
	- for $j=1$ to $k$ (recompute the cluster centers/means/$\vec{m}$'s):
		- $m_j \leftarrow \frac{1}{|C_j|} \sum_{\vec{x}\in C_j} \vec{x}$ (for each cluster, calculate the mean and update)

Note that doing the two key `for` operations above effectively decreases the $J_{\text{avg}}^2$ loss because each step either reassigns points to closer centers or updates centers to minimize within-cluster variance.

Remarks:
- Likely the most popular clustering algorithm
- Effectively does alternating minimization on $J_{\text{avg}}^2 = \sum_{j=1}^{k} \sum_{x \in C_j} d(x,m_j)^2$
- Converges in a finite number of steps because (1) only finite number of partitions of the $n$ samples into $k$ clusters and (2) the loss keeps decreasing until convergence. 
	- A partition of the $n$ samples into $k$ clusters is a way of dividing the dataset into $k$ non-overlapping, non-empty subsets such that each sample is only in one cluster, and one cluster has at least one sample. You have $n$ finite samples. Each sample can only be assigned to one of $k$ clusters. So there are only finitely many ways to assign each sample to a cluster.
	- But the worst-case no. steps can still be VERY large: $O(n^{kd})$ especially in high dimensions or with adversarial inputs
- Finding the optimal clustering is NP-hard for general $p$ (even for $k = 2$) or general $k$ (even $d = 2$)
	- As we often either manipulate the no. of features or the no. of clusters
	- The number of possible partitions is exponential.
	- The objective is non-convex with many local minima.
	- Even verifying if a solution is optimal requires checking all partitions — which is infeasible as $n$, $k$, or $d$ increase.
- No guarantee to converge to the globally optimal solution (in most cases it won’t) $\rightarrow$ very serious problem. Often end up with some clusters only having a single datapoint. Solutions:
	- Random restarts
	- Merge clusters that are too small
	- Split clusters that are too large
	- Annealing and other methods for dealing with complicated energy surfaces

### Hiarchecical clustering
- Input: the datapoints $x_1, x_2, \ldots, x_n \in \mathbb{R}^d$  
- Output: a clustering tree (dendrogram)

Advantages:
- Don’t need to decide number of clusters in advance
- Hierarchical structure is often very informative

Types:
- Agglomerative (bottom-up): start with $n$ clusters containing one datapoint each, and then merge clusters pairwise until only one cluster is left.
- Divisive (top-down): start with a single cluster containing all the datapoints and then split it into smaller and smaller clusters. → Recursively cluster clusters into smaller clusters.

Merging criteria for agglomerative clustering: Agglomerative algorithms always merge the pair of clusters closest to each other according to some distance measure:
- Single linkage: $d(C_i, C_j) = \min_{x \in C_i, x' \in C_j} d(x, x')$  
  → tends to generate long “chains”
- Complete linkage: $d(C_i, C_j) = \max_{x \in C_i, x' \in C_j} d(x, x')$  
  → tends to generate compact “round” clusters, $k$–center cost
- Average linkage:  $d(C_i, C_j) = \frac{1}{|C_i|} \frac{1}{|C_j|} \sum_{x \in C_i} \sum_{x' \in C_j} d(x, x')$. 
	- Ward’s method → $k$–means cost of resulting clustering

Agglomerative clustering algorithm:
```
C = set()
for i in range(1, n + 1):
    C.add({x[i]})  # Each point starts as its own cluster

while len(C) > 1:
    # Find the pair of clusters C1, C2 with smallest d(C1, C2)
    C1, C2 = find_closest_pair(C)
    C.remove(C1)
    C.remove(C2)
    C.add(C1.union(C2))  # Merge C1 and C2
```

### Model-based clustering: Gaussian Mixture Model
Each data point is a pair $(\vec{x}_i, z_i)$ in which 
- $\vec{x}_i$ is a $d$-dim vector of the $i$-th sample ($d$ is the number of features): is observed
- $z_i \in {1\cdots K}$ is the cluster assignment of the $i$-th data point: is hidden. 
The key assumption is that each $(\vec{x}_i, z_i)$ is drawn independently from some probability distribution with parameters $\theta$ or $(\vec{x}_i, z_i) \sim \Pr_\theta$. In other words, the probability distribution $\Pr_\theta$ can generate "data". This is a probabilistic model.

In this particular case, we use the Gaussian model to estimate the parameters. That is, 
$$\Pr(\vec{x})=\mathcal{N}(\vec{x}|\vec{\mu},\Sigma)=\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}e^{-(-(x-\mu)^T\Sigma^{-1}(x-\mu)/2)}$$
Some key terms to clarify:
* $z$ is in range $1\cdots K$ which are the cluster IDs and there are $K$ clusters. $z$ represents a Gaussian cluster. Remember, this is hidden/unknown before we do anything.
* $\pi_z = \Pr(z=1)$ is the probability of seeing cluster $z$ and $\Sigma_{j=1}^K = 1$.
* $\vec{\mu}_z \in \mathbb{R}^{d}$ is the mean/center of cluster $z$. Note: the vector sign is to make sure this is not a scalar, although this is a single point in the $d$-dim vector space
* $\Sigma_z \in \mathbb{R}^{d\times d}$ is the covariance matrix of cluster $z$. Note that, it's not $\sigma$ because we work with $\geq 2$-dim.
* $\Pr(\vec{x}_i, z) =\mathcal{N}(\vec{x}_i | \vec{\mu}_z, \Sigma_z)$ is the probability  of seeing $\vec{x}_i$ if $\vec{x}_i$ already in the Gaussian cluster $z$
* Paremeters to care about: $\pi$, $\mu$, $\Sigma$

So, the marginal probability of a sample, $\vec{x}$, is $$\Pr(\vec{x})=\mathcal{N}(\vec{x}|\vec{\mu}, \Sigma) = \sum_{j=1}^K \pi_j\mathcal{N}(\vec{x}|\vec{\mu}_j, \Sigma_j)$$
For each sample, we want to find the parameters $\theta = (\pi, \mu, \Sigma)$ such that they make $\Pr(\vec{x}$) as high as possible (i.e. more chances to be into "correct" or "similar" cluster). We use statistical estimation like Maximum Likelihood Estimation (MLE) to attempt to do so. 

So, given dataset $\textbf{X}$ of shape $N \times D$ and of independent samples, we want to maxmize the following probability, $$\Pr(\textbf{X}|\pi, \vec{\mu}, \Sigma) = \prod_{i=1}^{N} \left[ \sum_{j=1}^K \pi_j\mathcal{N}(\vec{x}_i|\vec{\mu}_j,\Sigma_j) \right]$$
The above probability means that, given the set of parameters (3 of them), the probability of seeing $\textbf{X}$ depends on how each of the samples is being placed into its cluster $j$ among (1 to K) correctly. Intuitively, for each sample $\vec{x}_i$, the model computes a weighted sum of probabilities over all $K$ clusters. Each term in the sum reflects the likelihood that $\vec{x}_i$ was generated by Gaussian cluster $j$. Finally, the product just simply shows that these samples are independent and we're allowed to multiply its probability (after the summing step). 

Now, two important things:

First, suppose, $z_{i,j} = 1$ if $\vec{x_i}$ is in cluster $z_k$. So, $\Pr (z_{i,k} = 1 | \vec{x}_i)$ is the probability of seeing cluster $z_k$ contains $\vec{x}_i$ or not, given $\vec{x}_i$. Bayes theorem to expand the term: $$\Pr(z_{i,k}=1|\vec{x}_i)=\frac{\Pr(z_{k})\cdot\Pr(\vec{x}_i|z_{i,k}=1)}{\Pr(\vec{x}_i)} = \frac{\pi_k\cdot \Pr(\vec{x}_i|\vec{\mu}_k,\Sigma_k)}{\sum_{j=1}^K \pi_j\mathcal{N}(\vec{x}|\vec{\mu}_j, \Sigma_j)}$$ and also, each point belongs to each cluster **with some probability** and GMM is soft clustering (meaning each data point can belong to multiple clusters at the same time, with degrees of membership (probabilities or weights), instead of being assigned to just one cluster.). Thus, we define this term: $$N_k=\sum_{i=1}^N \Pr(z_{i,k} =1 |\vec{x}_i)$$ This term means the expected number of points in cluster $k$ - a real number, not necessarily an integer.

Second, to maximize the $\Pr(\textbf{X}|\pi, \mu, \Sigma)$, we simply need to find the derivatives w.r.t $\pi, \mu, \Sigma$, and then set them to 0, and compute these parameters such that $\Pr(\textbf{X}|\pi, \mu, \Sigma)$ has the optimal solution (max). So, look at,
$$\Pr(\textbf{X}|\pi, \vec{\mu}, \Sigma) = \prod_{i=1}^{N} \left[ \sum_{j=1}^K \pi_j\mathcal{N}(\vec{x}_i|\vec{\mu}_j,\Sigma_j) \right] = \prod_{i=1}^{N} \left[ \sum_{j=1}^K \pi_j \Pr(\vec{x}_i) \right] = \prod_{i=1}^{N} \left[ \sum_{j=1}^K \pi_j \frac{1}{\sqrt{(2\pi)^d|\Sigma_j|}}e^{-(-(\vec{x}_i-\vec{\mu}_j)^T\Sigma_j^{-1}(\vec{x}_i-\vec{\mu}_j)/2)} \right]$$
This function is non-convex because it involves a sum of exponentials inside a product, leading to a log-likelihood that is the sum of log-sum-exp terms, which are non-convex in the parameters $\pi_j$, $\vec{\mu}_j$, and $\Sigma_j$. Thus, we **transform it using logarithm and turning this problem into a log-likelihood maximization**. Not writing everything down here, we get the following expressions for the parameters given generic cluster $k$:
$$\mu_k=\frac{1}{N_k}\sum_{i=1}^N\Pr(z_{i,k}=1|\vec{x}_i)\cdot\vec{x}_i$$

$$\Sigma_k=\frac{1}{N_k} = \sum_{i=1}^N\Pr(z_{i,k}=1|\vec{x}_i)(\vec{x}_i-\vec{\mu}_k)(\vec{x}_i-\vec{\mu}_k)^T$$

$$\pi_k=\frac{N_k}{N}$$
There is no closed-form solution, so we use the Expectation-Maximization (EM) algorithm to iteratively find a local maximum of the log-likelihood.

What is useful to see about the above two set of expressions is that: For a cluster $k$, 
- Parameters $\vec{\mu}_k, \Sigma_k, \pi_k \rightarrow \Pr(z_{i,k}=1|\vec{x}_i)$ (the bayes in first thing)
- $\Pr(z_{i,k}=1|\vec{x}_i) \rightarrow \vec{\mu}_k, \Sigma_k, \pi_k$ parameters (the derivative in second thing)

Since MLE can't be used because one of the variables ($z$) is hidden (we don't know the cluster assignment procedure yet before doing anything!), we can opt for Expectation Maximization or EM algorithm in which we can simply calculate the expected likelihood w.r.t the hidden variable $z$ without needing to know about it yet (initially). 

The worded EM algorithm:
1. Initialize $\theta=(\mu_k, \Sigma_k, \pi_k)$
2. E-Step: compute expected likelihoods w.r.t hidden variable $z$. Or calculate $\Pr_{\theta}(z_{i,k}=1|\vec{x}_i)$ from the 3 paremeters in $\theta$
3. M-Step: recompute the $\mu_k, \Sigma_k, \pi_k$ params to maximize the expected likelihoods. This means $(\mu_k, \Sigma_k, \pi_k)_{new} = \arg\max_{(\mu_k, \Sigma_k, \pi_k)} \Pr_{\theta}(z_{i,k}=1|\vec{x}_i)$. Repeat step 2 until convergence which means the change in log-likelihood or parameters falls below a threshold, or the log-likelihood no longer increases significantly.

GMM is better than K-means because it does not assume the same-size clusters. 