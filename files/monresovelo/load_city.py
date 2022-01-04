import osmnx as ox
import os
import pickle

G = ox.graph_from_point((45.516955, -73.567614), 185)

print(len(G.nodes))

if not os.path.exists(os.path.dirname("files/monresovelo/city_graphs/city.ox")):
    os.makedirs(os.path.dirname("files/monresovelo/city_graphs/city.ox"))

with open("files/monresovelo/city_graphs/city.ox", "wb") as outfile:
    pickle.dump(G, outfile)