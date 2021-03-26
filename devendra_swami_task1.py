import sys
import json
import networkx as nx


if __name__ == '__main__':
    infile = sys.argv[1]
    gexf_outfile = sys.argv[2]
    json_outfile = sys.argv[3]

    # read_tweets
    tweets = []
    with open(infile, "r", encoding="utf-8") as f:
        for line in f.readlines():
            temp = json.loads(line)
            tweets.append(temp)


    # create graph
    G = nx.DiGraph() 

    for tweet in tweets:
        from_user = tweet["user"]["screen_name"]
        if "retweeted_status" in tweet:
            to_user = tweet["retweeted_status"]["user"]["screen_name"]
            if G.has_edge(from_user, to_user):
                G[from_user][to_user]["weight"] += 1
            else:
                G.add_weighted_edges_from([(from_user, to_user, 1.0)])
        else:
            G.add_node(from_user)

    # Save graph
    nx.write_gexf(G,gexf_outfile)

    # Prepare output
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    max_retweeted_user, max_retweeted_number = sorted(list(G.in_degree(weight='weight')), key=lambda item: item[1], reverse=True)[0]
    max_retweeter_user, max_retweeter_number = sorted(list(G.out_degree(weight='weight')), key=lambda item: item[1], reverse=True)[0]

    output = {"n_nodes": n_nodes, "n_edges": n_edges, "max_retweeted_user": max_retweeted_user, "max_retweeted_number": max_retweeted_number, "max_retweeter_user": max_retweeter_user, "max_retweeter_number": max_retweeter_number}

    with open(json_outfile,"w") as f:
        json.dump(output,f)