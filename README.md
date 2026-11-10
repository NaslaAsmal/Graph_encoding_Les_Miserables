# Graph_encoding_Les_Miserables
Graph Representation Learning Using Random Walk Embedding 
# ASSIGNMENT 2
# GRAPH REPRESENTATION LEARNING USING DEEPWALK (RANDOM WALK + WORD2VEC)

**Submitted by:** Nasla K  
**Register Number:** 2026PHD21005 
**Course:** DSC622 Graph Algorithms and Mining  
**Programming Language:** Python  
**Environment:** Anaconda Jupyter Notebook  

---

## 1. INTRODUCTION

Graph representation learning is an important area of machine learning that focuses on converting graph-structured data into numerical representations called embeddings. Graphs consist of nodes and edges, where nodes represent entities and edges represent relationships between them.

Traditional machine learning algorithms generally require numerical input features. However, graph data contains complex relationships that cannot always be represented directly using conventional feature extraction methods.

In this assignment, I implemented a graph representation learning technique called **DeepWalk** using the Les Misérables character network dataset. DeepWalk uses random walks to generate sequences of connected nodes and applies the Word2Vec Skip-gram model to learn low-dimensional numerical representations of the nodes.

The implementation was carried out using Python in Anaconda Jupyter Notebook.

## 2. OBJECTIVES

The main objectives of this assignment are:

1. To understand the concept of graph representation learning.
2. To study the working principle of random walks in graph networks.
3. To implement the DeepWalk algorithm using Python.
4. To generate random-walk sequences from a graph dataset.
5. To learn node embeddings using the Word2Vec Skip-gram model.
6. To visualize the learned node embeddings using Principal Component Analysis (PCA).
7. To identify similarities between nodes using cosine similarity.

## 3. DATASET DESCRIPTION

For this assignment, I selected the **Les Misérables Character Co-occurrence Network**, which is available as a built-in graph dataset in the NetworkX library.

The dataset represents relationships between characters appearing in Victor Hugo's novel *Les Misérables*.

In this network:

- Each node represents a character in the novel.
- Each edge represents a co-occurrence relationship between two characters.
- Edge weights represent the frequency of co-occurrence.

### Dataset Statistics

| Property | Value |
|---|---|
| Dataset name | Les Misérables |
| Number of nodes | 77 |
| Number of edges | 254 |
| Graph type | Undirected |
| Data source | NetworkX built-in dataset |

The dataset was loaded directly using the NetworkX library without requiring an external download.

## 4. THEORETICAL BACKGROUND

### 4.1 Graph Representation Learning

Graph representation learning is the process of converting nodes in a graph into numerical vectors. These vectors are known as node embeddings.

The purpose of embeddings is to represent the structural relationships between nodes in a numerical format that can be used by machine learning algorithms.

In this assignment, each character in the Les Misérables network is represented using a 64-dimensional numerical vector.

### 4.2 Random Walk

A random walk is a process in which a traversal starts from a particular node and moves to one of its neighbouring nodes randomly.

The process continues until a predefined walk length is reached.

For example, a random walk may generate a sequence such as:

Jean Valjean → Cosette → Marius → Éponine

The sequence represents a possible path through the character network.

Multiple random walks are generated from each node to capture different structural relationships within the graph.

### 4.3 DeepWalk

DeepWalk is a graph representation learning algorithm that combines random walks with the Word2Vec Skip-gram model.

The algorithm consists of two main stages:

1. Random walks are performed on the graph to generate node sequences.
2. Word2Vec is applied to these sequences to learn numerical representations of the nodes.

In DeepWalk, nodes are treated as words, while random-walk sequences are treated as sentences.

The Skip-gram model learns node representations by predicting surrounding context nodes from a given node.

### 4.4 Word2Vec Skip-gram

Word2Vec is a technique used to learn numerical representations from sequences.

The Skip-gram architecture attempts to predict the surrounding context of a given input node.

In this implementation, Word2Vec learns relationships between characters based on their occurrence within random-walk sequences.

### 4.5 Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique used to reduce the number of features while preserving as much of the original data variance as possible.

Since the DeepWalk model generates 64-dimensional embeddings, PCA is used to reduce them to two dimensions for visualization.

### 4.6 Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them.

It is used in this assignment to identify characters with similar learned embeddings.

## 5. SOFTWARE AND LIBRARIES USED

The implementation was performed using the following Python libraries:

| Library | Purpose |
|---|---|
| NetworkX | Loading and handling the graph |
| NumPy | Numerical operations |
| Pandas | Organizing embedding values |
| Matplotlib | Graph and embedding visualization |
| Gensim | Word2Vec implementation |
| Scikit-learn | PCA and cosine similarity |
| Random | Generating random walks |

## 6. METHODOLOGY AND IMPLEMENTATION

The implementation was carried out in the following stages.

### Step 1: Importing the Required Libraries

The necessary Python libraries were imported to perform graph operations, generate random walks, train the embedding model and visualize the results.

### Step 2: Loading the Dataset

The Les Misérables graph was loaded using:

`G = nx.les_miserables_graph()`

The graph contains 77 nodes and 254 edges.

The number of nodes and edges was displayed to understand the structure of the dataset.

### Step 3: Exploring and Visualizing the Graph

The graph was examined to understand its structure, connectivity and relationships between characters.

The NetworkX spring layout was used to visualize the character network. Each node represents a character, and the edges represent their co-occurrence relationships.

**Figure 1: Les Misérables Character Network**

[Insert screenshot of the original graph visualization]

### Step 4: Implementing Random Walk

A random walk function was implemented to generate sequences of connected nodes.

The function accepts three parameters:

- Graph: The input network.
- Start node: The node from which the walk begins.
- Walk length: The number of nodes in the generated sequence.

At each step, one neighbouring node is randomly selected and added to the walk.

### Step 5: Generating Multiple Random Walks

Multiple random walks were generated from every node in the network.

The following parameters were selected:

| Parameter | Value |
|---|---:|
| Number of walks per node | 10 |
| Walk length | 20 |
| Total nodes | 77 |
| Total random walks | 770 |

The generated sequences were stored in a list and used as the training input for Word2Vec.

### Step 6: Training the Word2Vec Model

The generated random-walk sequences were supplied to the Word2Vec model.

The Skip-gram architecture was selected to learn node embeddings.

The following parameters were used:

| Parameter | Value |
|---|---:|
| Embedding dimension | 64 |
| Context window | 5 |
| Minimum count | 0 |
| Architecture | Skip-gram |
| Number of epochs | 20 |
| Workers | 1 |

The model learned a numerical representation for each character based on its surrounding context in the random-walk sequences.

### Step 7: Extracting Node Embeddings

After training the Word2Vec model, embeddings were extracted for all 77 characters.

Each character was represented by a vector containing 64 numerical values.

Therefore, the complete embedding matrix had the dimensions:

**77 × 64**

The embeddings were organized using a Pandas DataFrame and saved as a CSV file.

### Step 8: Applying PCA

PCA was applied to reduce the 64-dimensional embeddings into two dimensions.

The original embedding matrix had the dimensions 77 × 64, while the reduced representation had the dimensions 77 × 2.

This made it possible to visualize all character embeddings in a two-dimensional coordinate system.

### Step 9: Visualizing the Learned Embeddings

A scatter plot was generated using the two principal components obtained from PCA.

Each point in the visualization represents one character.

Characters positioned close to one another have relatively similar learned representations in the embedding space.

**Figure 2: Two-Dimensional Visualization of DeepWalk Embeddings**

[Insert screenshot of the PCA embedding visualization]

### Step 10: Similarity Analysis

Cosine similarity was used to compare the learned embeddings of characters.

The implementation calculated the similarity between a selected character and all other characters in the network.

The characters were ranked according to their cosine similarity scores.

This analysis demonstrates how learned embeddings can be used to identify structurally similar nodes.

## 7. RESULTS AND OBSERVATIONS

The DeepWalk implementation successfully generated node embeddings for the Les Misérables character network.

The main experimental observations are summarized below:

| Experiment | Result |
|---|---|
| Nodes in the graph | 77 |
| Edges in the graph | 254 |
| Random walks generated | 770 |
| Walk length | 20 |
| Embedding dimension | 64 |
| Total node embeddings | 77 |
| PCA dimensions | 2 |

The PCA visualization provided a two-dimensional representation of the learned embeddings.

The scatter plot showed the relative distribution of characters in the embedding space. Characters with similar network contexts could be observed in nearby regions.

The cosine similarity experiment provided a numerical comparison between character embeddings.

The learned embeddings were also exported into a CSV file for further analysis.

## 8. DISCUSSION

Through this implementation, I understood how graph representation learning can convert complex graph structures into numerical representations.

Random walks helped generate sequences of connected nodes, capturing local structural relationships within the network.

Word2Vec then processed these sequences and learned embeddings based on the contextual relationships between characters.

The use of PCA helped visualize the learned representations in two dimensions, making it easier to examine the distribution of characters.

The experiment demonstrated that DeepWalk can learn node representations without manually designing feature vectors for individual nodes.

However, the quality of the embeddings depends on parameters such as walk length, number of walks, context window and embedding dimensions.

The random walks used in this implementation select neighbouring nodes uniformly and do not use the original edge weights.


## 9. CONCLUSION

In this assignment, I successfully implemented graph representation learning using the DeepWalk algorithm on the Les Misérables character co-occurrence network.

The implementation involved loading the graph using NetworkX, generating random-walk sequences, training a Word2Vec Skip-gram model and extracting 64-dimensional node embeddings.

PCA was used to reduce the embeddings to two dimensions for visualization, while cosine similarity was used to compare the learned representations of characters.

Through this assignment, I gained practical understanding of random walks, Word2Vec, node embeddings and dimensionality reduction.

The experiment demonstrated how graph-structured data can be converted into numerical representations using a combination of random walks and language-model-based learning techniques.
