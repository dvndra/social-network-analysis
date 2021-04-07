# Social-Network-Analysis
[![Supported Python version](http://dswami.freevar.com/git_icons/pyversions.svg)](https://www.python.org/downloads/)

**DSCI 553: Foundations and Applications of Data Mining**

## Assignment Overview
In this assignment, you will use twitter data along with networkx and gephi to
form social networks and do analysis by implementing a community detection algorithm on the created network. You will also explore topological methods for network representation such as matrix representation of graph,
graph Laplacian, and spectral methods.

### Task 1: Creating Retweet Network and Analyzing it

For this task, you will create a retweet network and perform analysis on it. 
Retweet network is a weighted and directed graph in which nodes are
representing users and there are edges between users based on the retweet
relation (retweet relation == X retweets Y). The edges between two users
should be weighted (if the retweet relation holds between two same users
increase the weight). The edges should also be directed based on the
retweet relation meaning that if “X retweets Y” then there should be an
edge from X (the source) to Y (the target).

For Task 1 you need to answer the following questions:
- Given a json file create the retweet network for it and save the network as a gexf.
- How many nodes does this network have? 
- How many edges does this network have? 
- Which user’s tweets get the most retweets? We need the screen name of this
user. 
- What is the number of retweets the above user received?
- Which user retweets the most? We need the screen name of this user. 
- What is the number of retweets the above user did?

Input format:
`python firstname_lastname_task1.py <input_filename> <gexf_output_filename> <json_output_filename>`
Params: 
- input_file_name: the name of the input file 
- gexf_output_file_name: The name of the output gexf file, including file path
- json_ouutput_file_name: The name of the output JSON file, including file path

### Task 2: Community Detection

For this task, you will be implementing the CLAN algorithm discussed in class to detect communities on the gamergate dataset (on the same network you created in the previous task). CLAN is a two-step community detection method that uses node attributes to debias previous community detection methods that tend to create too many singleton communities without taking content of the node into consideration. For the first step of CLAN, you will implement an unsupervised community detection method (the Grivan-Newman algorithm). For the second step, you will train a classifier to use node attributes in the graph on the major communities detected by Grivan-Newman and classify non-significant communities into one of those major communities.

For step 1 of CLAN, you need to perform the following tasks:
- Partition the graph into communities that maximize the modularity objective.
- You should also consider the singleton communities as a valid community.
- Save your results as a txt file where the first line reports the modularity value of the best split and the following lines are the detected communities where each line represents one community with the following format:
***‘node1_Screen_name’, ‘node2_Screen_name’, ‘node3_Screen_name’,...***
- Sort the results based on the community size in ascending order and then the first node screen name in the community in lexicographical order. The screen names of nodes in each community should also be in the lexicographical order. 

For step 2 of CLAN, you need to perform the following tasks:
- Take the two largest communities detected in step 1 and train a Multinomial Naïve Bayes classifier based on TFIDF features of the nodes in the detected largest communities and text used by them (remember each node is a user and each user has a set of tweets they tweeted) 
- Take all the tweets for a specific user and use TFIDF features on that text to train the classifier. 
- After training the classifier, classify each of the nodes from the other smaller communities to one of these significant communities. Imagine these users in small communities as a test instance that you are labeling them based on their features (text tweeted by them) on a classifier that was trained on nodes from the two significant communities. 
- Create a txt file with the two communities and the nodes in them following the same format as in step 1 except you do not need to report the modularity score this time in the first line. 

Step 3: As further modification, train a Multinomial Naïve Bayes classifier this time using the count-vectorizer features and repeat the task in step 2. Create a new txt file and report the communities in it following the same format as in step 2 of CLAN.

Input format:
```python firstname_lastname_task2.py <input_filename> <step1_output_filename> <step2_output_filename> <step3_output_filename>```
Param: 
- input_file_name: The name of the input json file, including file path.
- taskA_output_file_name: The name and path of the output txt file for task A.
- taskB_output_file_name: The name and path of the output txt file for task B.
- taskC_output_file_name: The name and path of the output txt file for task C.

### Task 3 : k-way spectral graph partition on a large graph

Now, let’s apply the spectral graph partition algorithm on some real-world data, for example, a graph of email communication. (https://snap.stanford.edu/data/email-Eu-core.html). The dataset is included in the data folder. It includes two files, email-Eu-core.txt_ contains edge information, each line represents an edge between two nodes. File email-Eu-core-department-labels.txt_ contains node label information, each line specifies the label of a node, 42 labels in total. They are ground truth cluster labels.

Task: Clustering a large graph into k clusters.

Input format:
```python firstname_lastname_task_3.py <edge_filename> <output_filename> <k>```

Params:
- edge_filename : Path to edge file, e.g.: data/email-Eu-core.txt
- output_filename : Path to the the output file
- k: Number of clusters, e.g.: 42

Output format: Similar to email-Eu-core-department-labels.txt, the second
column should be predicted cluster label. The absolute value of the labels doesn’t matter.

### Task 4 : Node classification based on spectral embedding

Task: Perform node classification based on learned spectral embedding. A sample train/test split is provided as labels_train.csv/labels_test.csv

Input format:
```python firstname_lastname_task_4.py <edge_filename> <label_train_filename> <label_test_filename> <output_filename>```

Params:
- edge_filename : Path to edge file, e.g.: data/email-Eu-core.txt
- label_train_filename : Path to train label file
- label_test_filename: Path to test label file
- output_filename : Path to the output file

Output format: Similar to labels_test_truth.csv, the second column should be
your model’s prediction.

### Task 5: Identify important nodes in a graph via page rank

Page rank is initially used to find important webpages, but it generalizes to find important nodes in any type of graph, for example, important persons in an email communication graph. The same hypothesis in page rank could be used in email communication: if a lot of emails are sending to a person, then that person is very likely to be important. Getting an email from an important person makes you more likely to be an important person.

Task: Find the 20 most important nodes in a graph via page rank. Use random teleport with beta = 0. 8 and always teleport for dead ends.

Input format:
``` python firstname_lastname_task_6.py <edge_filename> <output_filename>```

Params:
- edge_filename : Path to edge file, e.g.: data/email-Eu-core.txt
- output_filename : Path to output file

Output format: A text file contains node index of the 20 most important nodes, sorted by their importance in decreasing order, one node index per line.

