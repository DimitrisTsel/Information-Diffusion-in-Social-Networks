#!/usr/bin/env python
# coding: utf-8

# information diffusion in social networks

# In[]:
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import operator as op
# In[]


file=r'C:\Users\30694\Desktop\Xalkidi\facebook_combined.txt'
G=nx.read_edgelist(file, create_using=nx.Graph, edgetype=int)
nx.info(G)

# In[]:


#calculate metrices of G
clos_cen=nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)
bet_cen = nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
eig_cen = nx.eigenvector_centrality(G,max_iter=100)
deg=dict(nx.degree(G))
clust_coef=nx.clustering(G)


# In[]:


#calculate the avg metrices of G
avg_clos=round((sum(clos_cen.values())/len(clos_cen)),4)
avg_bet=round((sum(bet_cen.values())/len(bet_cen)),4)
avg_eig=round((sum(eig_cen.values())/len(eig_cen)),4)
avg_deg=round((sum(deg.values())/len(deg)),4)
cluster=nx.average_clustering(G)
cluster_avg=round(cluster,4)
print('Average Degree: '+ str(avg_deg))
print('Average Closeness Centrality: '+ str(avg_clos))
print('Average Betweeness Centrality: '+ str(avg_bet))
print('Average Clustering Coefficient: '+ str(cluster_avg))
print('Average Eigenvector Centrality: '+ str(avg_eig))


# In[]:


#top 10 nodes by degree
sorted_degree = sorted(deg.items(),key=op.itemgetter(1), reverse=True)
print("Top 10 nodes by degree:")
for d in sorted_degree[:10]:
    print(d)


# In[]:


#top 10 nodes by betweeness
sorted_betweenness = sorted(bet_cen.items(),key=op.itemgetter(1), reverse=True)

print("Top 20 nodes by betweenness:")
for b in sorted_betweenness[:10]:
    print(b)


# In[]:


#top 10 nodes by closeness
sorted_closeness = sorted(clos_cen.items(),key=op.itemgetter(1), reverse=True)

print("Top 10 nodes by closeness:")
for b in sorted_closeness[:10]:
    print(b)


# In[]:


#top 10 nodes by eigenvector centrality
sorted_eigen = sorted(eig_cen.items(),key=op.itemgetter(1), reverse=True)

print("Top 20 nodes by eigenvector centrality:")
for b in sorted_eigen[:10]:
    print(b)


# In[]:


#top 10 nodes by clustering coefficient
sorted_clust_coef = sorted(clust_coef.items(),key=op.itemgetter(1), reverse=True)

print("Top 10 nodes by clustering coefficient:")
for b in sorted_clust_coef[:10]:
    print(b)


# In[]:


fig, ax = plt.subplots(figsize=(100,80))
nx.draw(G, with_labels=True, node_color='lightblue', alpha=0.5, ax=ax,)


# Independent Cascade Model

# In[]:


# Network topology
G1 = nx.erdos_renyi_graph(1000, 0.02)
G2 = nx.erdos_renyi_graph(1000, 0.07)
G3 = nx.erdos_renyi_graph(1000, 0.2)
G4 = nx.erdos_renyi_graph(1000, 0.5)


# In[]:


# Calculate graph Metrics
def calculate_metrics(G):
    avg_degree = round(pd.DataFrame(G.degree()).iloc[:,1].mean(), 4)
    avg_clustering_coefficient = round(nx.average_clustering(G), 4)
    avg_closeness = round(pd.DataFrame(nx.closeness_centrality(G).items()).iloc[:,1].mean(), 4)
    avg_betweeness = round(pd.DataFrame(nx.betweenness_centrality(G).items()).iloc[:,1].mean(), 4)
    avg_eigenvector = round(pd.DataFrame(nx.eigenvector_centrality(G).items()).iloc[:,1].mean(), 4)
    return avg_degree, avg_clustering_coefficient, avg_closeness,avg_betweeness,avg_eigenvector


# In[]:


m1 = calculate_metrics(G1)
m2 = calculate_metrics(G2)
m3 = calculate_metrics(G3)
m4 = calculate_metrics(G4)


# In[]:


# Independent Cascade Model for random seed
def independent_cascade_random_seed (G):
    # Model selection
    model = ep.IndependentCascadesModel(G)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('fraction_infected', 0.01)

    # Setting the edge parameters
    threshold = 0.1
    for e in G.edges():
        config.add_edge_configuration("threshold", e, threshold)
    

    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(20)
    return(iterations)


# In[]:


#Independent Cascade for graphs
df1 = pd.DataFrame(independent_cascade_random_seed(G1))
df2 = pd.DataFrame(independent_cascade_random_seed(G2))
df3 = pd.DataFrame(independent_cascade_random_seed(G3))
df4 = pd.DataFrame(independent_cascade_random_seed(G4))


# In[]:


# Independent Cascade for random seed
g1 = []
g2 = []
g3 = []
g4 = []

for i in range (0,20):
    g1.append(df1['node_count'][i].get(2))
    g2.append(df2['node_count'][i].get(2))
    g3.append(df3['node_count'][i].get(2))
    g4.append(df4['node_count'][i].get(2))

plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1:avg_deg: " + str(m1[0]) + ", avg_clust: " + str(m1[1]) + ", avg_clos: " + str(m1[2]) + ", avg_bet: "+str(m1[3])+ ", avg_eig: " + str(m1[4]))
plt.plot(df2["iteration"], g2, 'g', label = "G2:avg_deg: " + str(m2[0]) + ", avg_clust: " + str(m2[1]) + ", avg_clos: " + str(m2[2])+ ", avg_bet: "+str(m2[3])+ ", avg_eig: " + str(m2[4]))
plt.plot(df3["iteration"], g3, 'b', label = "G3:avg_deg: " + str(m3[0]) + ", avg_clust: " + str(m3[1]) + ", avg_clos: " + str(m3[2])+ ", avg_bet: "+str(m3[3])+ ", avg_eig: " + str(m3[4]))
plt.plot(df4["iteration"], g4, 'c', label = "G4:avg_deg: " + str(m4[0]) + ", avg_clust: " + str(m4[1]) + ", avg_clos: " + str(m4[2])+ ", avg_bet: "+str(m4[3])+ ", avg_eig: " + str(m4[4]))

plt.legend(loc='best', prop={'size': 8})
plt.title('Independent Cascade for random seed')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


# In[]:


#max degree
deg1 = pd.DataFrame(nx.degree(G1))
deg2 = pd.DataFrame(nx.degree(G2))
deg3 = pd.DataFrame(nx.degree(G3))
deg4 = pd.DataFrame(nx.degree(G4))

max_deg1 = list(deg1.nlargest(10, [1]).index)
max_deg2 = list(deg2.nlargest(10, [1]).index)
max_deg3 = list(deg3.nlargest(10, [1]).index)
max_deg4 = list(deg4.nlargest(10, [1]).index)


# In[]:


#max clustering coefficient
cl_coef1 = pd.DataFrame(nx.clustering(G1).items())
cl_coef2 = pd.DataFrame(nx.clustering(G2).items())
cl_coef3 = pd.DataFrame(nx.clustering(G3).items())
cl_coef4 = pd.DataFrame(nx.clustering(G4).items())

max_cl_coef1 = list(cl_coef1.nlargest(10, [1]).index)
max_cl_coef2 = list(cl_coef2.nlargest(10, [1]).index)
max_cl_coef3 = list(cl_coef3.nlargest(10, [1]).index)
max_cl_coef4 = list(cl_coef4.nlargest(10, [1]).index)


# In[]:


#max betweenness
btwn1 = pd.DataFrame(nx.betweenness_centrality(G1).items())
btwn2 = pd.DataFrame(nx.betweenness_centrality(G2).items())
btwn3 = pd.DataFrame(nx.betweenness_centrality(G3).items())
btwn4 = pd.DataFrame(nx.betweenness_centrality(G4).items())

max_btwn1 = list(btwn1.nlargest(10, [1]).index)
max_btwn2 = list(btwn2.nlargest(10, [1]).index)
max_btwn3 = list(btwn3.nlargest(10, [1]).index)
max_btwn4 = list(btwn4.nlargest(10, [1]).index)


# In[]:


#max eigenvector
eig1 = pd.DataFrame(nx.eigenvector_centrality(G1).items())
eig2 = pd.DataFrame(nx.eigenvector_centrality(G2).items())
eig3 = pd.DataFrame(nx.eigenvector_centrality(G3).items())
eig4 = pd.DataFrame(nx.eigenvector_centrality(G4).items())

max_eig1 = list(eig1.nlargest(10, [1]).index)
max_eig2 = list(eig2.nlargest(10, [1]).index)
max_eig3 = list(eig3.nlargest(10, [1]).index)
max_eig4 = list(eig4.nlargest(10, [1]).index)

# In[]:


#max closeness
clos1 = pd.DataFrame(nx.closeness_centrality(G1).items())
clos2 = pd.DataFrame(nx.closeness_centrality(G2).items())
clos3 = pd.DataFrame(nx.closeness_centrality(G3).items())
clos4 = pd.DataFrame(nx.closeness_centrality(G4).items())

max_clos1 = list(clos1.nlargest(10, [1]).index)
max_clos2 = list(clos2.nlargest(10, [1]).index)
max_clos3 = list(clos3.nlargest(10, [1]).index)
max_clos4 = list(clos4.nlargest(10, [1]).index)
# In[]:


# Independent Cascade Model for given seed
def independent_cascade (G , seed):
    # Model selection
    model = ep.IndependentCascadesModel(G)
    
    # Model Configuration
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", seed)
        
    # Setting the edge parameters
    threshold = 0.1
    for e in G.edges():
        config.add_edge_configuration("threshold", e, threshold)
    
    model.set_initial_status(config)
    
    # Simulation execution
    iterations = model.iteration_bunch(20)
    return(iterations)


# In[]:


# Independent Cascade for seed of max degree
df_max_deg1 = pd.DataFrame(independent_cascade (G1 , max_deg1))
df_max_deg2 = pd.DataFrame(independent_cascade (G2 , max_deg2))
df_max_deg3 = pd.DataFrame(independent_cascade (G3 , max_deg3))
df_max_deg4 = pd.DataFrame(independent_cascade (G4 , max_deg4))


# In[]:


# Plot Independent Cascade Model for random seed vs seed of max degree
n1 = []
n2 = []
n3 = []
n4 = []

for i in range (0,20):
    n1.append(df_max_deg1["node_count"][i].get(2))
    n2.append(df_max_deg2["node_count"][i].get(2))
    n3.append(df_max_deg3["node_count"][i].get(2))
    n4.append(df_max_deg4["node_count"][i].get(2))
#graphs with random seed 
plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1 random seed")
plt.plot(df2["iteration"], g2, 'g', label = "G2 random seed")
plt.plot(df3["iteration"], g3, 'b', label = "G3 random seed")
plt.plot(df4["iteration"], g4, 'c', label = "G4 random seed")
#graphs with max degree seed
plt.plot(df_max_deg1["iteration"], n1,'r', linestyle='dashed' ,label='G1 max degree seed')
plt.plot(df_max_deg2["iteration"], n2,'g', linestyle='dashed' ,label='G2 max degree seed')
plt.plot(df_max_deg3["iteration"], n3,'b', linestyle='dashed' ,label='G3 max degree seed')
plt.plot(df_max_deg4["iteration"], n4,'c', linestyle='dashed' ,label='G4 max degree seed')
plt.title('Independent Cascade Model for random seed vs seed of max degree')
plt.legend(loc = 'best')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


# In[]:


#Independent Cascade for seed with max clustering coefficient
df_max_cl_coef1 = pd.DataFrame(independent_cascade (G1 , max_cl_coef1))
df_max_cl_coef2 = pd.DataFrame(independent_cascade (G2 , max_cl_coef2))
df_max_cl_coef3 = pd.DataFrame(independent_cascade (G3 , max_cl_coef3))
df_max_cl_coef4 = pd.DataFrame(independent_cascade (G4 , max_cl_coef4))


# In[]:


# Plot Independent Cascade Model for random seed vs seed with max clustering coefficient
n1 = []
n2 = []
n3 = []
n4 = []

for i in range (0,20):
    n1.append(df_max_cl_coef1["node_count"][i].get(2))
    n2.append(df_max_cl_coef2["node_count"][i].get(2))
    n3.append(df_max_cl_coef3["node_count"][i].get(2))
    n4.append(df_max_cl_coef4["node_count"][i].get(2))
#graphs with random seed
plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1 random seed")
plt.plot(df2["iteration"], g2, 'g', label = "G2 random seed")
plt.plot(df3["iteration"], g3, 'b', label = "G3 random seed")
plt.plot(df4["iteration"], g4, 'c', label = "G4 random seed")
#praphs with max clustering coefficient seed
plt.plot(df_max_cl_coef1["iteration"], n1,'r', linestyle='dashed' ,label='G1 max clus_coef seed')
plt.plot(df_max_cl_coef2["iteration"], n2,'g', linestyle='dashed' ,label='G2 max clus_coef seed')
plt.plot(df_max_cl_coef3["iteration"], n3,'b', linestyle='dashed' ,label='G3 max clus_coef seed')
plt.plot(df_max_cl_coef4["iteration"], n4,'c', linestyle='dashed' ,label='G4 max clus_coef seed')
plt.title('Independent Cascade Model for random seed vs seed with max clustering coefficient')
plt.legend(loc = 'best')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


# In[]:


#Independent Cascade for seed with max betweenness
df_max_btwn1 = pd.DataFrame(independent_cascade (G1 , max_btwn1))
df_max_btwn2 = pd.DataFrame(independent_cascade (G2 , max_btwn2))
df_max_btwn3 = pd.DataFrame(independent_cascade (G3 , max_btwn3))
df_max_btwn4 = pd.DataFrame(independent_cascade (G4 , max_btwn4))


# In[]:


# Plot Independent Cascade Model for random seed vs seed with max betweenness vertices
n1 = []
n2 = []
n3 = []
n4 = []

for i in range (0,20):
    n1.append(df_max_btwn1["node_count"][i].get(2))
    n2.append(df_max_btwn2["node_count"][i].get(2))
    n3.append(df_max_btwn3["node_count"][i].get(2))
    n4.append(df_max_btwn4["node_count"][i].get(2))
#graphs with random seed 
plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1 random seed")
plt.plot(df2["iteration"], g2, 'g', label = "G2 random seed")
plt.plot(df3["iteration"], g3, 'b', label = "G3 random seed")
plt.plot(df4["iteration"], g4, 'c', label = "G4 random seed")
#graphs with max betweenness seed 
plt.plot(df_max_btwn1["iteration"], n1,'r', linestyle='dashed' ,label='G1 max betweenness seed')
plt.plot(df_max_btwn2["iteration"], n2,'g', linestyle='dashed' ,label='G2 max betweenness seed')
plt.plot(df_max_btwn3["iteration"], n3,'b', linestyle='dashed' ,label='G3 max betweenness seed')
plt.plot(df_max_btwn4["iteration"], n4,'c', linestyle='dashed' ,label='G4 max betweenness seed')

plt.title('Independent Cascade Model for random seed vs seed with max betweenness')
plt.legend(loc = 'best')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


# In[]:


# Run Independent Cascade for seed with max eigenvector 
df_max_eig1 = pd.DataFrame(independent_cascade (G1 , max_eig1))
df_max_eig2 = pd.DataFrame(independent_cascade (G2 , max_eig2))
df_max_eig3 = pd.DataFrame(independent_cascade (G3 , max_eig3))
df_max_eig4 = pd.DataFrame(independent_cascade (G4 , max_eig4))


# In[]:


# Plot Independent Cascade Model for random seed vs seed with max eigenvector
n1 = []
n2 = []
n3 = []
n4 = []

for i in range (0,20):
    n1.append(df_max_eig1["node_count"][i].get(2))
    n2.append(df_max_eig2["node_count"][i].get(2))
    n3.append(df_max_eig3["node_count"][i].get(2))
    n4.append(df_max_eig4["node_count"][i].get(2))

#graphs with random seed
plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1 random seed")
plt.plot(df2["iteration"], g2, 'g', label = "G2 random seed")
plt.plot(df3["iteration"], g3, 'b', label = "G3 random seed")
plt.plot(df4["iteration"], g4, 'c', label = "G4 random seed")
#graphs with max eigenvector seed 
plt.plot(df_max_eig1["iteration"], n1,'r', linestyle='dashed' ,label='G1 max eigenvector seed')
plt.plot(df_max_eig2["iteration"], n2,'g', linestyle='dashed' ,label='G2 max eigenvector seed')
plt.plot(df_max_eig3["iteration"], n3,'b', linestyle='dashed' ,label='G3 max eigenvector seed')
plt.plot(df_max_eig4["iteration"], n4,'c', linestyle='dashed' ,label='G4 max eigenvector seed')

plt.title('Independent Cascade Model for random seed vs seed of max eigenvector vertices')
plt.legend(loc = 'best')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


# In[]:


# Run Independent Cascade for seed with max closeness 
df_max_clos1 = pd.DataFrame(independent_cascade (G1 , max_clos1))
df_max_clos2 = pd.DataFrame(independent_cascade (G2 , max_clos2))
df_max_clos3 = pd.DataFrame(independent_cascade (G3 , max_clos3))
df_max_clos4 = pd.DataFrame(independent_cascade (G4 , max_clos4))


# In[]:


# Plot Independent Cascade Model for random seed vs seed with max closeness
n1 = []
n2 = []
n3 = []
n4 = []

for i in range (0,20):
    n1.append(df_max_clos1["node_count"][i].get(2))
    n2.append(df_max_clos2["node_count"][i].get(2))
    n3.append(df_max_clos3["node_count"][i].get(2))
    n4.append(df_max_clos4["node_count"][i].get(2))

#graphs with random seed
plt.figure(figsize=(10,6))
plt.plot(df1["iteration"], g1, 'r', label = "G1 random seed")
plt.plot(df2["iteration"], g2, 'g', label = "G2 random seed")
plt.plot(df3["iteration"], g3, 'b', label = "G3 random seed")
plt.plot(df4["iteration"], g4, 'c', label = "G4 random seed")
#graphs with max eigenvector seed 
plt.plot(df_max_clos1["iteration"], n1,'r', linestyle='dashed' ,label='G1 max closeness seed')
plt.plot(df_max_clos2["iteration"], n2,'g', linestyle='dashed' ,label='G2 max closeness seed')
plt.plot(df_max_clos3["iteration"], n3,'b', linestyle='dashed' ,label='G3 max closeness seed')
plt.plot(df_max_clos4["iteration"], n4,'c', linestyle='dashed' ,label='G4 max closeness seed')

plt.title('Independent Cascade Model for random seed vs seed of max closeness vertices')
plt.legend(loc = 'best')
plt.xlabel('Number of Iterations')
plt.ylabel('Infected Nodes')
plt.show()


