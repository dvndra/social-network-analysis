import sys
import json
from copy import deepcopy
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
# import matplotlib.pyplot as plt
import pdb

def create_graph(infile):
    tweets = []
    user_tweet_doc = {}
    with open(infile, "r", encoding="utf-8") as f:
        for line in f.readlines():
            temp = json.loads(line)
            tweets.append(temp)


    G = nx.Graph() 
    
    for tweet in tweets:
        from_user = tweet["user"]["screen_name"]
        if from_user in user_tweet_doc:
            user_tweet_doc[from_user] = user_tweet_doc[from_user] + " " + tweet["text"]
        else:
            user_tweet_doc[from_user] = tweet["text"]

        if "retweeted_status" in tweet:
            to_user = tweet["retweeted_status"]["user"]["screen_name"]
            if to_user in user_tweet_doc:
                user_tweet_doc[to_user] = user_tweet_doc[to_user] + " " + tweet["retweeted_status"]["text"]
            else:
                user_tweet_doc[to_user] = tweet["retweeted_status"]["text"]
            if G.has_edge(from_user, to_user):
                G[from_user][to_user]["weight"] += 1      
            else:
                G.add_weighted_edges_from([(from_user, to_user, 1.0)])
        else:
            G.add_node(from_user)

    return G, user_tweet_doc

# print("Number of nodes in this digraph =", G.number_of_nodes())
# print("Number of edges in this digraph =", G.number_of_edges())

def edges_to_remove(G): 

    # Will return edges in the form [(a,b),(c,d) ....] , weight="weight"
    value_dict = nx.edge_betweenness_centrality(G, normalized=False, weight = "weight")  
    max_val = -1
    max_edges = []
    for key,val in value_dict.items():
        if val>max_val:
            max_edges = [key]
            max_val = val
        elif val==max_val:
            max_edges.append(key)
      
    return max_edges 


def calculate_modularity(G, comm, num_edges, node_list, node_list_degrees, node_index_dict):
    result = 0.0
    adj_matrix = nx.adjacency_matrix(G, nodelist = node_list, weight = "weight")
    # pdb.set_trace()
    for node_set in comm:
        for node1 in node_set:
            for node2 in node_set:
                if node1!=node2:
                    node1_index = node_index_dict[node1]
                    node2_index = node_index_dict[node2]
                    result += adj_matrix[node1_index, node2_index]
                    result -= ((node_list_degrees[node1_index][1]*node_list_degrees[node2_index][1])/float(2*num_edges))
                

    return result/(2*num_edges)


def girvan(G):
    
    # returns list of sets of screen_names
    comm = list(nx.connected_components(G)) 
    result = comm
    num_comp = len(comm)  
    
    node_list_degrees = list(G.degree(weight = "weight"))
    node_list = [x[0] for x in node_list_degrees]
    node_index_dict = dict(map(lambda t: (t[1], t[0]), enumerate(node_list)))
    num_edges = G.number_of_edges()
    max_modularity = 0
    max_edges = edges_to_remove(G)

    while (len(max_edges) > 0): 
        for edge in max_edges:
            u, v = edge 
            G.remove_edge(u, v)  
          
        comm = list(nx.connected_components(G)) 
        if len(comm) > num_comp:
            num_comp = len(comm) 
            curr_modularity = calculate_modularity(G, comm, num_edges, node_list, node_list_degrees, node_index_dict)
            if curr_modularity > max_modularity:
                max_modularity = curr_modularity
                result = comm
                # print(max_modularity)
        max_edges = edges_to_remove(G)   

    return result, max_modularity 


def make_predictions(X_train_counts,train_labels, X_test_counts, max_one, max_two, test_comm_list, outfile):
    # #train a multinomial naive Bayes classifier
    clf = MultinomialNB().fit(X_train_counts, train_labels)

    # #do the predictions
    predicted_labels = clf.predict(X_test_counts)

    comm_one = deepcopy(max_one)
    comm_two = deepcopy(max_two)
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==0:
            comm_one.append(test_comm_list[i])
        else:
            comm_two.append(test_comm_list[i])
    
    with open(outfile, 'w') as f:
        if(len(comm_one)<=len(comm_two)):
            f.write("'"+ "','".join(sorted(comm_one))+"'"+"\n")
            f.write("'"+ "','".join(sorted(comm_two))+"'"+"\n")
        else:
            f.write("'"+ "','".join(sorted(comm_two))+"'"+"\n")
            f.write("'"+ "','".join(sorted(comm_one))+"'"+"\n")


if __name__ == '__main__':

    # Task A
    infile = sys.argv[1]
    outfile_mod = sys.argv[2]
    outfile_tfidf = sys.argv[3]
    outfile_count = sys.argv[4]
    
    G, user_tweet_doc = create_graph(infile)
    communities, max_modularity = girvan(G)
    communities = [sorted(list(x)) for x in communities]
    communities = sorted(communities, key = lambda x: (len(x),x[0]))

    with open(outfile_mod, 'w') as f:
        f.write("Best Modularity is: " + str(max_modularity) + "\n")
        for community in communities:
            f.write("'"+ "','".join(community)+"'"+"\n")

    # Task B and C
    # Prepare data for ml model
    max_one = communities[-1]
    max_two = communities[-2]
    test_comm_list = [node for community in communities[:-2] for node in community]
    # print(len(max_one),len(max_two),len(test_comm_list))
    train_data = []
    train_labels = []
    test_data = []

    for node in max_one:
        train_data.append(user_tweet_doc[node])
        train_labels.append(0)
    for node in max_two:
        train_data.append(user_tweet_doc[node])
        train_labels.append(1)
    for node in test_comm_list:
        test_data.append(user_tweet_doc[node])

    # #extract the tfidf features and make predictions
    tf_idf_vect = TfidfVectorizer()
    X_train_counts = tf_idf_vect.fit_transform(train_data)
    X_test_counts = tf_idf_vect.transform(test_data)
    make_predictions(X_train_counts,train_labels, X_test_counts, max_one, max_two, test_comm_list, outfile_tfidf)
    
    # PART C
    # #extract the countvectorizer features for the train data
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(test_data)
    make_predictions(X_train_counts,train_labels, X_test_counts, max_one, max_two, test_comm_list, outfile_count)
    