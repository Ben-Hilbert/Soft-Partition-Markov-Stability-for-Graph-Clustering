import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib as plt
import random
import pandas as pd
import math
import osmnx as ox
import scipy.sparse as sp
import sklearn as sci
from scipy.linalg import qr
from scipy.linalg import logm
from scipy.linalg import svd
from scipy.linalg import polar
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.interpolate import griddata

#################################################
#Helper Functions 
def computeVk(k0,G,A,degrees):
    I = sparse.diags(np.ones(A.shape[0]))
    inv_sqrt_degrees=[1/np.sqrt(x) for x in degrees]
    D_inv_sqrt=sparse.diags(inv_sqrt_degrees)
    temp=D_inv_sqrt @ csr_matrix(nx.adjacency_matrix(G)) @ D_inv_sqrt
    LN = I - D_inv_sqrt @ csr_matrix(nx.adjacency_matrix(G)) @ D_inv_sqrt
    eigenvalues, eigenvectors = sparse.linalg.eigs(temp, k=k0, which="LM")
    Vk = eigenvectors[:, :k0]
    return Vk

def computeP(k,x,G,Vk):
    matrix = tf.zeros((k, k), dtype=x[0].dtype)
    # Fill the strictly upper triangular part of the matrix
    index = 0
    for i in range(k):
        for j in range(i + 1, k):
            matrix = tf.tensor_scatter_nd_update(matrix, [[i, j]], [x[index]])
            index += 1
    S=matrix-tf.transpose(matrix)
    Q=tf.linalg.expm(S)
    temp=Vk@Q
    return temp**2

def computeMSMat(P,graphMat):
    return tf.transpose(P)@graphMat@P

def computeInitialGuess(A,k):
    Q, R, P = qr(A, pivoting=True)
    if np.linalg.det(Q) < 0:
        # Swap the first and second columns of Q
        Q[:, [0, 1]] = Q[:, [1, 0]]
    S=logm(Q)
    index = 0
    initial = np.zeros(int((k*(k-1))/2))
    for i in range(k):
        for j in range(i + 1, k):
            initial[index]=S[i,j]
            index += 1
    return initial

def polar_factorization(A):
    # Compute the singular value decomposition
    U, s, Vh = svd(A)
    # Construct the diagonal matrix Sigma
    Sigma = np.diag(s)
    # Compute the positive semi-definite Hermitian matrix P
    P = Vh.T @ Sigma @ Vh
    # U is the unitary (orthogonal) matrix from the SVD
    return U, P

def computeInitialGuessPolar(matrix,k):
    Q, R, P = qr(matrix, pivoting=True)
    selectedCols=P[0:k]
    U,P=polar(matrix[:,selectedCols])
    if np.linalg.det(U) < 0:
        # Swap the first and second columns of Q
        U[:, [0, 1]] = U[:, [1, 0]]
    S=logm(U)
    index = 0
    initial = np.zeros(int((k*(k-1))/2))
    for i in range(k):
        for j in range(i + 1, k):
            initial[index]=S[i,j]
            index += 1
    return initial

def plotOutput(P,k,times,title):
    x_values=np.arange(P.shape[0])
    plt.pyplot.figure(figsize=(10, 6))
    for j in range(k):
        y_values=P[:,j]
        plt.pyplot.plot(x_values, y_values, label=f"Distribution {j+1}")
    plt.pyplot.xlabel('Node', fontsize=22) #20 for one pane
    plt.pyplot.ylabel('Weight', fontsize=22) #20 for one pane
    plt.pyplot.tick_params(axis='both', labelsize=18) #14 for one pane
    plt.pyplot.title(f'{title}: Optimal Solution for t={times[0]}', fontsize=20) #20 for one pane
    plt.pyplot.legend(loc="lower left",fontsize=18) #16 for one pane
    plt.pyplot.savefig(f"{title}.pdf")
    plt.pyplot.show()

def eps_connectivity_graph(eps=0.25,var=0.5,n=800,visual=False): #generates an epsilon connectivity graph for a given value of epsilon with 4 "clusters"
    #define 4 bivariate gaussian distributions
    mean1 = [0, 0]
    mean2 = [2, 0]
    mean3 = [0, 2]
    mean4 = [2, 2]
    cov = [[var, 0], [0, var]] #common covariance matrix
    #sample from the mixture distribution with uniform weights
    samples = np.random.multinomial(n, [0.25,0.25,0.25,0.25], 1)[0]
    data=[]
    temp1=np.random.multivariate_normal(mean1, cov, samples[0])
    for i in range(samples[0]):
        data.append(tuple(temp1[i]))
    temp2=np.random.multivariate_normal(mean2, cov, samples[1])
    for i in range(samples[1]):
        data.append(tuple(temp2[i]))
    temp3=np.random.multivariate_normal(mean3, cov, samples[2])
    for i in range(samples[2]):
        data.append(tuple(temp3[i]))
    temp4=np.random.multivariate_normal(mean4, cov, samples[3])
    for i in range(samples[3]):
        data.append(tuple(temp4[i]))
    x, y = zip(*data)
    #generate graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if (i!=j) and (math.dist(data[i],data[j])<eps):
                G.add_edge(i,j)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    largest_cc=list(largest_cc)
    initial_groups = np.concatenate([
    np.full(samples[0], 1),  # first n_1 entries are 1
    np.full(samples[1], 2),  # next n_2 entries are 2
    np.full(samples[2], 3),  # next n_3 entries are 3
    np.full(samples[3], 4)   # last n_4 entries are 4
    ])
    groups=np.zeros(len(largest_cc))
    xdat=np.zeros(len(largest_cc)) #data of the nodes in largest connected component
    ydat=np.zeros(len(largest_cc))
    for i in range(len(largest_cc)):
        groups[i]=initial_groups[largest_cc[i]]
        xdat[i]=x[largest_cc[i]]
        ydat[i]=y[largest_cc[i]]
    if(visual):  
        #coloring
        node_colors = ['orange'] * sum(groups==1) + ['blue'] * sum(groups==2)+['black'] * sum(groups==3) + ['green'] * sum(groups==4)
        pos = nx.spring_layout(G)  # For a better visual layout
        nx.draw(G, pos, with_labels=False, node_size=10, node_color=node_colors, edge_color="gray")
        plt.show()
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
    return G,xdat,ydat,groups

def maxMSObjGraphs(k,times,G,init_type,which,maxIter=50,tol=0.00001,gamma=-1): #init: identity, random, CPQR ... #which: Ithaca, Bay, Texas
        #create data frame
        dat = pd.DataFrame(columns=["Time","Markov Stability","Clustering"])

        if gamma==-1:
            gamma=1/(k-1)

        #precompute graph-related matrices
        scipy_sparse = sp.csr_matrix(nx.adjacency_matrix(G))
        A=tf.SparseTensor(
            indices=np.array(scipy_sparse.nonzero()).T,
            values=scipy_sparse.data,
            dense_shape=scipy_sparse.shape
        )
        n=G.number_of_nodes()
        degrees = np.ravel(nx.adjacency_matrix(G)@np.ones(n))
        #degrees_sparse=np.where(degrees < 3, 0, degrees)
        D=sparse.diags(degrees)
        inv_degrees=[1/x for x in degrees]
        T=sparse.diags(inv_degrees)@scipy_sparse
        m=G.number_of_edges()
        statDist=[x/(2*m) for x in degrees]
        Tt=T.copy()
        for step in range(times[0]-1):
            Tt=Tt@T
        prevT=Tt
        Vk=np.load(f"Modified_{which}_Vk{k}_data.npy")
        #initialize variables
        if init_type=="identity":
            vals=np.zeros(int((k*(k-1))/2))
        if init_type=="random":
            vals=np.random.rand(int((k*(k-1))/2))
        if init_type=="CPQR":
            vals=computeInitialGuessPolar(np.transpose(Vk),k)
        Vk=tf.convert_to_tensor(Vk,dtype=tf.float64)
        for i in range(len(times)):
            if i!=0:
                Tt=prevT@np.linalg.matrix_power(T,times[i]-times[i-1])
                #copy code from below if we ever use more times
            else:
                print("Optimizing")
                x=[tf.Variable(initial_value=vals[i]
                    , name=f'var_{i}',dtype=tf.float64) for i in range(int((k*(k-1))/2))]
                
                def objectiveFunction(x):
                    P=computeP(k,x,G,Vk)
                    allones=np.ones(n)
                    term1=tf.linalg.trace(tf.convert_to_tensor((np.transpose(P)@Tt)@P))
                    term2=tf.tensordot(np.transpose(P)@allones,np.transpose(P)@sp.diags(statDist)@allones,axes=1)
                    return -(term1-term2)
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                oldVal=0
                objective_values=np.zeros(maxIter)
                for step in range(maxIter):
                    print(f"Optimizing step: {step}")
                    with tf.GradientTape() as tape:
                        # Compute the negative objective function
                        loss = objectiveFunction(x)
                    
                    # Compute gradients
                    gradients = tape.gradient(loss, x)
                    
                    # Apply gradients to variables
                    optimizer.apply_gradients(zip(gradients, x))
                    oldVal=loss.numpy()
                    objective_values[step]=-objectiveFunction(x)
                PStar=computeP(k,x,G,Vk)
                ms=-objectiveFunction(x)
                res={"Time":times[i],"Markov Stability":ms.numpy(),"Clustering":PStar}
                #dat=dat.append(res,ignore_index=True)
        return  res, PStar, objective_values
#################################################

#Algorithm
#################################################
def maxMS(k,times,G,maxIter=500,tol=0.00001,gamma=-1): #times is an array of integer time values in ascending order
    #create data frame
    dat = pd.DataFrame(columns=["Time","Markov Stability","Clustering"])

    if gamma==-1:
        gamma=1/(k-1)

    #precompute graph-related matrices
    scipy_sparse = sp.csr_matrix(nx.adjacency_matrix(G))
    A=tf.SparseTensor(
        indices=np.array(scipy_sparse.nonzero()).T,
        values=scipy_sparse.data,
        dense_shape=scipy_sparse.shape
    )
    n=G.number_of_nodes()
    degrees = np.ravel(nx.adjacency_matrix(G)@np.ones(n))
    #degrees_sparse=np.where(degrees < 3, 0, degrees)
    D=sparse.diags(degrees)
    inv_degrees=[1/x for x in degrees]
    T=sparse.diags(inv_degrees)@scipy_sparse
    m=G.number_of_edges()
    statDist=[x/(2*m) for x in degrees]
    Tt=T.copy()
    for step in range(times[0]-1):
        Tt=Tt@T
    prevT=Tt
    Vk=computeVk(k,G,A,degrees)
    #initialize variables
    vals=computeInitialGuessPolar(np.transpose(Vk),k)
    Vk=tf.convert_to_tensor(Vk,dtype=tf.float64)
    for i in range(len(times)):
        if i!=0:
            Tt=prevT@np.linalg.matrix_power(T,times[i]-times[i-1])
            #copy code from below if we ever use more times
        else:
            print("Optimizing")
            x=[tf.Variable(initial_value=vals[i]
                , name=f'var_{i}',dtype=tf.float64) for i in range(int((k*(k-1))/2))]
            
            def objectiveFunction(x):
                P=computeP(k,x,G,Vk)
                allones=np.ones(n)
                term1=tf.linalg.trace(tf.convert_to_tensor((np.transpose(P)@Tt)@P))
                term2=tf.tensordot(np.transpose(P)@allones,np.transpose(P)@sp.diags(statDist)@allones,axes=1)
                return -(term1-term2)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
            oldVal=0
            for step in range(maxIter):
                print(f"Optimizing step: {step}")
                with tf.GradientTape() as tape:
                    # Compute the negative objective function
                    loss = objectiveFunction(x)
                
                # Compute gradients
                gradients = tape.gradient(loss, x)
                
                # Apply gradients to variables
                optimizer.apply_gradients(zip(gradients, x))
                if step>0:
                    percChange=np.absolute(np.absolute(loss.numpy()-oldVal)/oldVal)
                    if percChange<tol and step>1:
                        break
                oldVal=loss.numpy()
            PStar=computeP(k,x,G,Vk)
            ms=-objectiveFunction(x)
            res={"Time":times[i],"Markov Stability":ms.numpy(),"Clustering":PStar}
            print(PStar)
            print(ms)
    return  res, PStar
#################################################
#Data analysis
#srun --pty --partition=damle-interactive --time=20:00:00 --mem=80GB -N 1 -n 8 --gres=gpu:a6000:1 bash

#################################################
#SPMS Optimization on Standard SBM - Figure 4.1
if False:
    n_clusters = 4
    n_nodes_per_cluster = 100
    n_total_nodes = n_clusters * n_nodes_per_cluster

    # Define the sizes of the clusters
    sizes = [n_nodes_per_cluster] * n_clusters

    # Define the probability matrix for intra- and inter-cluster edges
    p_intra = 0.7  # Probability of edges within clusters
    p_inter = 0.1  # Probability of edges between clusters

    # Create the probability matrix
    p_matrix = np.full((n_clusters, n_clusters), p_inter)
    np.fill_diagonal(p_matrix, p_intra)

    # Generate the stochastic block model
    G = nx.stochastic_block_model(sizes, p_matrix)

    k=4
    times=[5]
    dat,P=maxMS(k,times,G,gamma=0)
    plotOutput(P,k,times,"4-Block SBM")

    #exact recovery?
    clustering=np.argmax(P,axis=1)
    print(clustering)
#################################################

#################################################
#SPMS Optimization on SBM with Ambiguous Nodes - Figure 4.2
if False:
    n_clusters = 4
    n_nodes_per_cluster = 100
    n_total_nodes = n_clusters * n_nodes_per_cluster

    # Define the sizes of the clusters
    sizes = [n_nodes_per_cluster] * n_clusters

    # Define the probability matrix for intra- and inter-cluster edges
    p_intra = 0.7  # Probability of edges within clusters
    p_inter = 0.1  # Probability of edges between clusters

    # Create the probability matrix
    p_matrix = np.full((n_clusters, n_clusters), p_inter)
    np.fill_diagonal(p_matrix, p_intra)

    #ensure the graph is connected
    while(True):
    # Generate the stochastic block model
        G = nx.stochastic_block_model(sizes, p_matrix)

        #Add noisy nodes
        noisy_node_start = n_total_nodes
        noisy_nodes = range(noisy_node_start, noisy_node_start + 100)

        # Add the noisy nodes to the graph
        G.add_nodes_from(noisy_nodes)

        # Connect noisy nodes with small uniform probability to all other nodes
        for noisy_node in noisy_nodes:
            for node in G.nodes:
                if noisy_node != node and np.random.rand() < 0.01:
                    G.add_edge(noisy_node, node)
        if nx.is_connected(G):
            break

    pos = nx.kamada_kawai_layout(G)

    # Highlight clusters with different colors
    colors = ['red', 'blue', 'green', 'purple']
    for i, color in enumerate(colors):
        nodes = range(i * n_nodes_per_cluster, (i + 1) * n_nodes_per_cluster)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=30, node_color=color, node_shape="o")

    # Highlight noisy nodes with a different color
    nx.draw_networkx_nodes(G, pos, nodelist=noisy_nodes, node_size=30, node_color="orange", node_shape="s")
    #Draw Edges
    nx.draw_networkx_edges(G, pos, edge_color="gray")

    plt.pyplot.savefig("Graph Visualization.pdf") #output is cropped for final manuscript
    plt.pyplot.show()

    k=4
    times=[5]
    dat,P=maxMS(k,times,G,gamma=0)
    plotOutput(P,k,times,"4-Block SBM with Ambiguous Nodes")
#################################################

#################################################
#SPMS Optimization on Gaussian Random Geometric Graphs - Figure 4.3
if False:
    num=100000
    G,xdat,ydat,counts=eps_connectivity_graph(eps=0.22,var=0.4,n=num,visual=False)
    k=4
    times=[5]
    dat,P=maxMS(k,times,G,gamma=0)
    color_map = {
                1: "orange",
                2: "blue",
                3: "black",
                4: "green"
            }
    colors = [color_map[group] for group in counts]

    #plots with data
    for i in range(4):
        p_alg=P[:,i]
        X, Y = np.meshgrid(np.linspace(min(xdat), max(xdat), 10000), np.linspace(min(ydat), max(ydat), 10000))
        Z = griddata((xdat, ydat), p_alg, (X, Y), method='cubic')
        fig = plt.pyplot.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7,vmin=0,vmax=max(p_alg))
        ax.set_xlabel('X', fontsize=20) #16
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Mass', fontsize=20, rotation=90)
        ax.tick_params(axis='x', labelsize=18) #12
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='z', labelsize=18)
        ax.set_title('Plot of Optimal Distributions for RGGs', fontsize=20)
        cbar=plt.pyplot.colorbar(surf)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.offsetText.set_fontsize(18)
        plt.pyplot.savefig(f"RGG_n={num}_No_Data_Dist_{i}.pdf")
        plt.pyplot.show()

    #plots without data
    for i in range(4):
        p_alg=P[:,i]
        X, Y = np.meshgrid(np.linspace(min(xdat), max(xdat), 10000), np.linspace(min(ydat), max(ydat), 10000))
        Z = griddata((xdat, ydat), p_alg, (X, Y), method='cubic')
        fig = plt.pyplot.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7,vmin=0,vmax=max(p_alg))
        scatter = ax.scatter(xdat, ydat, p_alg,  c=colors, alpha=0.6)
        ax.set_xlabel('X', fontsize=20) #16
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Mass', fontsize=20, rotation=90)
        ax.tick_params(axis='x', labelsize=18) #12
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='z', labelsize=18)
        ax.set_title('Plot of Optimal Distributions for RGGs with Data', fontsize=20)
        cbar=plt.pyplot.colorbar(surf)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.offsetText.set_fontsize(18)
        plt.pyplot.savefig(f"RGG_n={num}_With_Data_Dist_{i}.pdf")
        plt.pyplot.show()
#################################################

#################################################
#SPMS Optimization on Bay Area Road Network
#Plot of San Jose Optimal Distribution - Figure 4.4
if False:
    k=3
    nodes = pd.read_csv('modified_nodes_bay.csv')
    P=np.load(f"Modified_Bay_Optimal_k_{k}.npy")
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    p_alg=P[:,2]
    #make plots
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    plt.pyplot.scatter(xdat, ydat, c=p_alg, cmap='viridis', s=2, linestyle='None', marker='o')
    cbar=plt.pyplot.colorbar(label='Probability')
    cbar.set_label('Probability', size=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    plt.pyplot.title('Bay Area Optimal Distribution',fontsize=20)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.show()
    plt.pyplot.savefig(f"Optimal_Distribution_3_k{k}_Modified_Bay.pdf")

#Plot of Bay Area Hard Partition - Figure 4.4
if False:
    k=3
    nodes = pd.read_csv('modified_nodes_bay.csv')
    P=np.load(f"Modified_Bay_Optimal_k_{k}.npy")
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    clustering=np.argmax(P,axis=1)
    color_map = {0: "red",1: "blue",2: "green"}
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    cols=[color_map[i] for i in clustering]
    plt.pyplot.scatter(xdat, ydat, c=cols, s=2, marker='o')
    plt.pyplot.title('Bay Area Road Network Hard Partition',fontsize=20)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.savefig(f"Hard_Partition_k{k}_Modified_Bay.pdf")
    plt.pyplot.show()
#################################################

#################################################
#SPMS Optimization on Ithaca Road Network
#Plot of Binghamton Optimal Distribution - Figure 4.5
if False:
    k=2
    nodes = pd.read_csv('modified_nodes_ithaca.csv')
    P=np.load(f"Modified_Ithaca_Optimal_k_{k}.npy")
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    p_alg=P[:,1]
    #make plots
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    plt.pyplot.scatter(xdat, ydat, c=p_alg, cmap='viridis', s=2, linestyle='None', marker='o')
    cbar=plt.pyplot.colorbar(label='Probability')
    cbar.set_label('Probability', size=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    plt.pyplot.title('Ithaca Optimal Distribution',fontsize=22)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.show()
    plt.pyplot.savefig(f"Optimal_Distribution_k{k}_Modified_Ithaca.pdf")

#Plot of Ithaca Hard Partition - Figure 4.5
if False:
    k=2
    nodes = pd.read_csv('modified_nodes_ithaca.csv')
    P=np.load(f"Modified_Ithaca_Optimal_k_{k}.npy")
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    clustering=np.argmax(P,axis=1)
    color_map = {0:"red",1:"blue",2:"green"}
    #make plots
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    plt.pyplot.scatter(xdat, ydat, c=[color_map[i] for i in clustering], s=2, linestyle='None', marker='o')
    plt.pyplot.title('Ithaca Road Network Hard Partition',fontsize=22)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.show()
    plt.pyplot.savefig(f"Hard_Partition_k{k}_Modified_Ithaca.pdf")
#################################################

#################################################
#SPMS Optimization on Texas Road Network
#Plot of Dallas Optimal Distribution - Figure 4.6
if False:
    nodes = pd.read_csv('modified_nodes_texas.csv')
    P=np.load("Modified_Texas_Optimal_k_2.npy")
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    p_alg=P[:,0]
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    plt.pyplot.scatter(xdat, ydat, c=p_alg, cmap='viridis', s=2, linestyle='None', marker='o')
    cbar=plt.pyplot.colorbar(label='Probability')
    cbar.set_label('Probability', size=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    plt.pyplot.title('Texas Optimal Distribution',fontsize=20)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.show()
    plt.pyplot.savefig("Optimal_Distribution_1_k2_Modified_Texas.pdf")

#Plot of Texas Hard Partition - Figure 4.6
if False:
    nodes = pd.read_csv('modified_nodes_texas.csv')
    ydat=nodes.iloc[:,2]
    xdat=nodes.iloc[:,3]
    ind=nodes.index
    clustering=np.argmax(P,axis=1)
    color_map = {0:"red",1:"blue",2:"green"}
    #make plots
    fig,ax=plt.pyplot.subplots(figsize=(8, 6))
    plt.pyplot.scatter(xdat, ydat, c=[color_map[i] for i in clustering], s=2, linestyle='None', marker='o')
    plt.pyplot.title('Texas Road Network Hard Partition',fontsize=20)
    ax.grid(True,alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.pyplot.show()
    plt.pyplot.savefig("Hard_Partition_k2_Modified_Texas.pdf")
#################################################

#################################################
#Plot of SPMS Value by Iteration for Several Initializations on Road Networks - Figure 4.7
if False:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    GIth = nx.read_graphml('modified_ithaca_graph.graphml')
    GBay = nx.read_graphml('modified_bay_graph.graphml')
    GTex = nx.read_graphml('modified_texas_graph.graphml')

    times=[5]
    it=60

    dat_id_ith,P_id_ith,obj_id_ith=maxMSObjGraphs(2,times,GIth,init_type="identity",which="Ithaca",maxIter=it,gamma=0)
    dat_rand_ith,P_rand_ith,obj_rand_ith=maxMSObjGraphs(2,times,GIth,init_type="random",which="Ithaca",maxIter=it,gamma=0)
    dat_cpqr_ith,P_cpqr_ith,obj_cpqr_ith=maxMSObjGraphs(2,times,GIth,init_type="CPQR",which="Ithaca",maxIter=it,gamma=0)

    dat_id_bay,P_id_bay,obj_id_bay=maxMSObjGraphs(3,times,GBay,init_type="identity",which="Bay",maxIter=it,gamma=0)
    dat_rand_bay,P_rand_bay,obj_rand_bay=maxMSObjGraphs(3,times,GBay,init_type="random",which="Bay",maxIter=it,gamma=0)
    dat_cpqr_bay,P_cpqr_bay,obj_cpqr_bay=maxMSObjGraphs(3,times,GBay,init_type="CPQR",which="Bay",maxIter=it,gamma=0)

    dat_id_tex,P_id_tex,obj_id_tex=maxMSObjGraphs(2,times,GTex,init_type="identity",which="Texas",maxIter=it,gamma=0)
    dat_rand_tex,P_rand_tex,obj_rand_tex=maxMSObjGraphs(2,times,GTex,init_type="random",which="Texas",maxIter=it,gamma=0)
    dat_cpqr_tex,P_cpqr_tex,obj_cpqr_tex=maxMSObjGraphs(2,times,GTex,init_type="CPQR",which="Texas",maxIter=it,gamma=0)

    #plot relative to converged
    if False:
        def relative_to_converged(a):
            return (a-a[len(a)-1])/np.absolute(a[len(a)-1])

        obj_id_ith=relative_to_converged(obj_id_ith)
        obj_rand_ith=relative_to_converged(obj_rand_ith)
        obj_cpqr_ith=relative_to_converged(obj_cpqr_ith)

        obj_id_bay=relative_to_converged(obj_id_bay)
        obj_rand_bay=relative_to_converged(obj_rand_bay)
        obj_cpqr_bay=relative_to_converged(obj_cpqr_bay)

        obj_id_tex=relative_to_converged(obj_id_tex)
        obj_rand_tex=relative_to_converged(obj_rand_tex)
        obj_cpqr_tex=relative_to_converged(obj_cpqr_tex)

    fig, ax = plt.pyplot.subplots(figsize=(7, 4.8))
    plt.pyplot.subplots_adjust(right=0.8)
    ax.scatter(range(it), obj_id_ith, color='blue', s=10, marker='None', label='Identity')
    ax.plot(range(it), obj_id_ith, color='blue', linestyle='-', linewidth=2)
    ax.scatter(range(it), obj_rand_ith, color='red', s=10, marker='None', label='Random')
    ax.plot(range(it), obj_rand_ith, color='red', linestyle='-', linewidth=2)
    ax.scatter(range(it), obj_cpqr_ith, color='green', s=10, marker='None', label='CPQR')
    ax.plot(range(it), obj_cpqr_ith, color='green', linestyle='-', linewidth=2)

    ax.scatter(range(it), obj_id_bay, color='blue', s=10, marker='None', label='Identity')
    ax.plot(range(it), obj_id_bay, color='blue', linestyle='--', linewidth=2)
    ax.scatter(range(it), obj_rand_bay, color='red', s=10, marker='None', label='Random')
    ax.plot(range(it), obj_rand_bay, color='red', linestyle='--', linewidth=2)
    ax.scatter(range(it), obj_cpqr_bay, color='green', s=10, marker='None', label='CPQR')
    ax.plot(range(it), obj_cpqr_bay, color='green', linestyle='--', linewidth=2)

    ax.scatter(range(it), obj_id_tex, color='blue', s=10, marker='None', label='Identity')
    ax.plot(range(it), obj_id_tex, color='blue', linestyle=':', linewidth=2)
    ax.scatter(range(it), obj_rand_tex, color='red', s=10, marker='None', label='Random')
    ax.plot(range(it), obj_rand_tex, color='red', linestyle=':', linewidth=2)
    ax.scatter(range(it), obj_cpqr_tex, color='green', s=10, marker='None', label='CPQR')
    ax.plot(range(it), obj_cpqr_tex, color='green', linestyle=':', linewidth=2)

    inits=["Identity","Random","CPQR"]
    colors=["blue","red","green"]
    datasets=["Ithaca","Bay","Texas"]
    linestyles=["-","--",":"]

    init_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                    markersize=7, label=init) for init, color in zip(inits, colors)]

    legend1 = ax.legend(handles=init_handles, title='Initialization', bbox_to_anchor=(1, 0), loc='center left')
    ax.add_artist(legend1)

    dataset_handles = [mlines.Line2D([], [], color='black', marker="None", linestyle=ls,
                    markersize=7, label=dataset) for dataset, ls in zip(datasets, linestyles)]

    ax.legend(handles=dataset_handles, title='Dataset', bbox_to_anchor=(1, 0.25), loc='center left')

    ax.set_xlabel('Optimization Step',fontsize=12)
    ax.set_ylabel('SPMS',fontsize=12)
    plt.pyplot.xticks(fontsize=12)
    plt.pyplot.yticks(fontsize=12)
    ax.set_title('SPMS by Iteration with Varied Initialization',fontsize=12)
    plt.pyplot.savefig("SPMS_by_Iteration_3_Initializations.pdf")
    plt.pyplot.show()
#################################################

#################################################
#Core-Periphery Functions
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def flowBetweenFast(Start,End,Tt,G,p,t): #flow of probability between subsets Start -> End with distribution p
    return (p[Start]@(Tt[Start][:,End])@p[End]).item()

def get_neighbors(C,G):
    ret=[]
    for node in C:
        neigh=list(G.neighbors(node))
        for temp in neigh:
            if (temp not in C) and (temp not in ret):
                ret=np.append(ret,temp)
    return ret

def adjacent_core(p,n,G):
    Core=n_largest_indices(p,1)
    for i in range(1,n):
        neighbors=get_neighbors(Core,G) #all nodes adj to core but not core
        dictionary={}
        for j in neighbors:
            dictionary[j]=p[int(j)]
        Core = np.append(Core,max(dictionary, key=dictionary.get))
    return Core.astype(int)

def adjacent_core_seeded(p,G,prevCore): #computes the next core based on previous
    neighbors=get_neighbors(prevCore,G) #all nodes adj to core but not core
    dictionary={}
    for j in neighbors:
        dictionary[j]=p[int(j)]
    Core = np.append(prevCore,max(dictionary, key=dictionary.get))
    return Core.astype(int)

def n_largest_indices(arr, n):
    # Get indices that would sort arr in descending order
    return np.argsort(arr)[::-1][:n]

def plotOutput(P,k,times,title):
    x_values=np.arange(P.shape[0])
    plt.figure(figsize=(10, 7))
    for j in range(k):
        y_values=P[:,j]
        plt.plot(x_values, y_values, label=f"Distribution {j+1}")
    plt.xlabel('Node', fontsize=23)
    plt.ylabel('Weight', fontsize=23)
    plt.tick_params(axis='both', labelsize=20)
    plt.title(f'{title}: Optimal Solution for t={times[0]}', fontsize=24)
    plt.legend(loc="upper right",fontsize=22)
    plt.savefig(f"{title}.pdf")
    plt.show()

def plot_by_C(arr,n,title):
    x_values=list(range(n))
    x_values=x_values
    plt.figure(figsize=(10, 7))
    plt.plot(x_values, arr)
    plt.xlabel('Core Size', fontsize=23)
    plt.ylabel('Metric Value', fontsize=23)
    plt.tick_params(axis='both', labelsize=20)
    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(18)
    plt.title(title, fontsize=24)
    plt.savefig(f"{title}.pdf")
    plt.show()

def corePeripheryMetricFast(Core,Periphery,Tt,G,p,t):
    num=flowBetweenFast(Core,Core,Tt,G,p,t)
    den=len(Core)+flowBetweenFast(Periphery,Core,Tt,G,p,t)
    return num/den

def metricCoreExhaustive(prevCore,G,p,t): #semi-exhaustive approach that finds the node that increases the metric the most
    full_set = set(range(G.number_of_nodes()))
    prevPeriphery = list(full_set - set(prevCore))
    old_metric=corePeripheryMetricFast(prevCore,prevPeriphery,G,p,t)
    differences=np.zeros(G.number_of_nodes())
    for n in range(G.number_of_nodes()): #computes the differences in metric value
        if n in prevCore:
            differences[n]=-83
        else:
            tempCore=prevCore
            tempCore=np.append(tempCore,n)
            tempPeriphery = list(full_set - set(tempCore))
            differences[n]=corePeripheryMetricFast(tempCore,tempPeriphery,G,p,t)-old_metric
    #the node we want to add will be the arg max of differences
    Core=np.append(prevCore,np.argmax(differences))
    return Core

def metricCoreExhaustiveInitialization(G,p,t):
    full_set = set(range(G.number_of_nodes()))
    vals=np.zeros(G.number_of_nodes())
    for n in range(G.number_of_nodes()):
        Core=[n]
        Periphery=list(full_set-set(Core))
        vals[n]=corePeripheryMetricFast(Core,Periphery,G,p,t)
    return [np.argmax(vals)]

def generate_sbm(num_nodes_block1, num_nodes_block2, p_intra_block1, p_intra_block2, p_inter_block):
    # Define sizes of the two blocks
    sizes = [num_nodes_block1, num_nodes_block2]
    
    # Define the probability matrix
    # p[i][j] is the probability of an edge between a node in block i and a node in block j
    p = [
        [p_intra_block1, p_inter_block],  # Block 1 internal and cross-block edges
        [p_inter_block, p_intra_block2]   # Block 2 internal and cross-block edges
    ]
    
    # Generate the SBM graph
    G = nx.stochastic_block_model(sizes, p)
    
    return G

def generate_test_graph(title="",visual=True):
    num_nodes_block1 = 100    # Number of nodes in Block 1
    num_nodes_block2 = 100   # Number of nodes in Block 2
    p_intra_block1 = 0.4     # Intra-block edge probability for Block 1
    p_intra_block2 = 0.05     # Intra-block edge probability for Block 2
    p_inter_block = 0.1     # Inter-block edge probability

    CP1=generate_sbm(num_nodes_block1, num_nodes_block2, p_intra_block1, p_intra_block2, p_inter_block)
    CP2=generate_sbm(num_nodes_block1, num_nodes_block2, p_intra_block1, p_intra_block2, p_inter_block)
    mapping = {i: i + 200 for i in range(200)}
    CP2=nx.relabel_nodes(CP2, mapping)
    G=nx.union(CP1,CP2)

    #add egdes between the graph with some fixed probability (say 0.005)
    for i in range(200):
        for j in range(200,400):
            if(random.uniform(0,1)<0.005):
                G.add_edge(i,j)

    k=2
    times=[5]
    dat,P=maxMS(k,times,G,gamma=0)
    if visual:
        plotOutput(P,k,times,title)

    p_alg=P[:,0]
    return G,p_alg

def visualize_core(Core,x,y,groups,goi):
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    groups2=groups.copy()
    for n in Core:
        groups2[n]=5
    color_map2 = {
        1: "gray",
        2: "gray",
        3: "gray",
        4: "gray",
        5: "red"
    }
    color_map2[goi]="orange"
    colors2 = [color_map2[group] for group in groups2]
    ax2.scatter(x, y,c=colors2)
    plt.xlim(np.min(x[groups2==goi]-0.1), np.max(x[groups2==goi])+0.1)
    plt.ylim(np.min(y[groups2==goi]-0.1), np.max(y[groups2==goi])+0.1)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tick_params(axis='both', labelsize=30)
    ax2.set_title(f'Core Size: {len(Core)}', fontsize=28)
    ax2.set_xlabel('X', fontsize=30)
    ax2.set_ylabel('Y', fontsize=30)
    plt.tight_layout()
    plt.savefig(f'Data Points with Core (Core Size: {len(Core)}).pdf')
    plt.show()

def visualize_data(x,y,groups,goi):
    color_map = {
        1: "gray",
        2: "gray",
        3: "gray",
        4: "gray"
    }
    color_map[goi]="orange"
    colors = [color_map[group] for group in groups]
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    plt.xlim(np.min(x[groups==goi]-0.1), np.max(x[groups==goi])+0.1)
    plt.ylim(np.min(y[groups==goi]-0.1), np.max(y[groups==goi])+0.1)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tick_params(axis='both', labelsize=30)
    ax1.scatter(x, y,c=colors)
    ax1.set_title('Data Points', fontsize=28)
    ax1.set_xlabel('X', fontsize=30)
    ax1.set_ylabel('Y', fontsize=30)
    plt.tight_layout()
    plt.savefig(f'Core Periphery RGG Data Points.pdf')
    plt.show()

def computeTt(G,times):
    scipy_sparse = sp.csr_matrix(nx.adjacency_matrix(G))
    A=tf.SparseTensor(
        indices=np.array(scipy_sparse.nonzero()).T,
        values=scipy_sparse.data,
        dense_shape=scipy_sparse.shape
    )
    n=G.number_of_nodes()
    degrees = np.ravel(nx.adjacency_matrix(G)@np.ones(n))
    D=sparse.diags(degrees)
    inv_degrees=[1/x for x in degrees]
    T=sparse.diags(inv_degrees)@scipy_sparse
    m=G.number_of_edges()
    statDist=[x/(2*m) for x in degrees]
    Tt=T.copy()
    for step in range(times[0]-1):
        Tt=Tt@T
    return Tt
#################################################

#################################################
#Generic Core-Periphery SBM Test - Figure 5.1
#Code to Generate Data
if False:
    n_trials=50
    optimal_sizes=np.zeros(n_trials)
    cpmatrix=np.zeros((200,n_trials))
    x_values=list(range(200))
    times=[5]
    for j in range(n_trials):
        if j==0:
            G,p_alg=generate_test_graph("Core-Periphery SBM")
            plt.show()
            plt.figure(figsize=(10, 6))
        else:
            G,p_alg=generate_test_graph("Core-Periphery SBM",visual=False)
        Tt=computeTt(G,times)
        p_alg=np.array(p_alg)
        testMetric=np.zeros(200)
        prevCore=np.array(n_largest_indices(p_alg,1))
        full_set = set(range(400))
        maxVal=-1
        bestCore=list(full_set)
        for i in range(200):
            if i==0:
                Core=np.array(n_largest_indices(p_alg,1))
            else:
                Core=np.array(adjacent_core_seeded(p_alg,G,prevCore))
            Periphery = np.array(list(full_set - set(Core)))
            testMetric[i]=corePeripheryMetricFast(Core,Periphery,Tt,G,p_alg,5)
            if testMetric[i]>maxVal:
                maxVal=testMetric[i]
                bestCore=Core
            prevCore=list(Core)
        optimal_sizes[j]=len(bestCore)
        cpmatrix[:,j]=testMetric
        plt.plot(x_values, testMetric)
    np.save("CPSBM_Core_Metric_Matrix.npy",cpmatrix)
    np.save("CPSBM_Optimal_Core_Sizes.npy",optimal_sizes)
    plt.xlabel('Core Size', fontsize=16)
    plt.ylabel('Metric Value', fontsize=16)
    plt.title("Core-Periphery Metric by Core Size", fontsize=18)
    plt.savefig(f'CPSBM Core-Periphery Metric for {n_trials} Trials')

#Code for Final Plot
if False:
    G,p_alg=generate_test_graph("Core-Periphery SBM")
    plt.show()
    times=[5]
    n_trials=50
    Tt=computeTt(G,times)
    p_alg=np.array(p_alg)
    testMetric=np.zeros(200)
    prevCore=np.array(n_largest_indices(p_alg,1))
    full_set = set(range(400))
    maxVal=-1
    bestCore=list(full_set)
    for i in range(200):
        if i==0:
            Core=np.array(n_largest_indices(p_alg,1))
        else:
            Core=np.array(adjacent_core_seeded(p_alg,G,prevCore))
        Periphery = np.array(list(full_set - set(Core)))
        testMetric[i]=corePeripheryMetricFast(Core,Periphery,Tt,G,p_alg,5)
        if testMetric[i]>maxVal:
            maxVal=testMetric[i]
            bestCore=Core
        prevCore=list(Core)
    plt.figure(figsize=(10, 7))
    plt.xlabel('Core Size', fontsize=23)
    plt.ylabel('Metric Value', fontsize=23)
    plt.tick_params(axis='both', labelsize=20)
    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(18)
    plt.title("Core-Periphery Metric by Core Size", fontsize=24)
    x_values=list(range(200))
    cpmatrix=np.load("CPSBM_Core_Metric_Matrix.npy")
    cpmatrix=cpmatrix[:,0:49]
    for i in range(49):
        plt.plot(x_values,cpmatrix[:,i],color="gray",alpha=0.3)
    plt.plot(x_values,testMetric,color="red",linewidth=2.5)
    plt.savefig(f'CPSBM Core-Periphery Metric for {n_trials} Trials.pdf')
#################################################

#################################################
#Core-Periphery on Gaussian Random Geometric Graphs
#Core Metric for Several Random Draws - Figure 5.2
if False:
    #Note that the data for this figure was generated in batches
    #Each batch was produced using the following code
    it=1000 #was 500 (2000)
    n_trials=50
    x_values=list(range(it))
    optimal_sizes=np.zeros(n_trials)
    cpmatrix=np.zeros((it,n_trials))
    num=5000 #was 2000 (10000)
    plt.figure(figsize=(10, 6))
    graph_sizes=np.zeros(n_trials)
    for j in range(n_trials):
        G,xdat,ydat,counts=eps_connectivity_graph(eps=0.22,var=0.2,n=num,visual=False) #old variance was 0.4, eps was 0.22
        graph_sizes[j]=G.number_of_nodes()
        k=4
        times=[5]
        Tt=computeTt(G,times)
        dat,P=maxMS(k,times,G,gamma=0)
        p_alg=P[:,0]
        p_alg=np.array(p_alg)
        group_of_interest=counts[np.argmax(p_alg)] #should identify the group we analyze
        metric=np.zeros(it)
        prevCore=np.array(n_largest_indices(p_alg,1))
        full_set = set(range(G.number_of_nodes()))
        maxVal=-1
        bestCore=list(full_set)
        for i in range(it):
            print(i)
            if i==0:
                Core=np.array(n_largest_indices(p_alg,1))
            else:
                Core=np.array(adjacent_core_seeded(p_alg,G,prevCore))
            Periphery = list(full_set - set(Core))
            metric[i]=corePeripheryMetricFast(Core,Periphery,Tt,G,p_alg,5)
            if metric[i]>maxVal:
                maxVal=metric[i]
                bestCore=Core
            prevCore=Core
        optimal_sizes[j]=len(bestCore)
        cpmatrix[:,j]=metric
        plt.plot(x_values, metric)
        np.save("Core_Metric_Batch_1.npy",cpmatrix)
    np.save("Optimal_Sizes_Batch_1.npy",optimal_sizes)
    plt.xlabel('Core Size', fontsize=16)
    plt.ylabel('Metric Value', fontsize=16)
    plt.title("Core-Periphery Metric by Core Size", fontsize=18)
    plt.savefig(f'Core-Periphery Metric on RGGs for {n_trials} Trials')
    plt.show()
    plt.hist(optimal_sizes,bins=10)
    plt.savefig(f'Optimal Size Histogram for RGGs: {n_trials} Trials')
    print(graph_sizes)

#Code to generate the figure for metric value
if False:
    dat=np.load("Core_Metric_Batch_2.npy")
    m=dat.shape[0] #number of data points per trial
    n=dat.shape[1] #number of trials
    x_values=list(range(m))
    plt.figure(figsize=(10, 7))
    plt.xlabel('Core Size', fontsize=23)
    plt.ylabel('Metric Value', fontsize=23)
    plt.tick_params(axis='both', labelsize=20)
    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(18)
    plt.title("Core-Periphery Metric by Core Size", fontsize=24)
    avg_metric=(1/n)*(dat@np.ones(n)) #find a proxy for the "average" trial
    sum_of_metric=np.transpose(dat)@np.ones(m)
    imax=np.argmax(sum_of_metric) #proxy for an extreme trial - large case
    imin=np.argmin(sum_of_metric) #proxy for an extreme trial - small case
    square_dist=np.zeros(n)
    for i in range(n):
        xi=dat[:,i]
        square_dist[i]=np.dot(xi-avg_metric,xi-avg_metric)
    iavg=np.argmin(square_dist) #proxy for the trial closest to average
    special_indices=set([imax,imin,iavg])
    for i in range(n):
        if i not in special_indices:
            plt.plot(x_values, dat[:,i],color="gray",alpha=0.3)
    for j in special_indices:
        if j==iavg:
            plt.plot(x_values,dat[:,j],color="red",linewidth=2.5)
        else:
            plt.plot(x_values,dat[:,j],color="red",linestyle=":",linewidth=2.5)
    plt.savefig(f'Core-Periphery Metric on RGGs for {n} Trials.pdf')

#Code to create histogram of optimal core sizes
if False:
    dat=np.load("Optimal_Sizes_Batch_2.npy")
    n=len(dat)
    plt.figure(figsize=(10, 7))
    plt.xlabel('Core Size', fontsize=23)
    plt.ylabel('Frequency', fontsize=23)
    plt.tick_params(axis='both', labelsize=20)
    plt.title("Optimal Core Sizes", fontsize=24)
    plt.hist(dat,bins=12,edgecolor="black",alpha=0.7,color="slateblue")
    plt.tight_layout()
    plt.savefig(f'Optimal Core Sizes on RGGs for {n} Trials.pdf')
    plt.show()

#Plot of Core Iterations - Figure 5.3
if False:
    G,xdat,ydat,counts=eps_connectivity_graph(eps=0.22,var=0.4,n=5000,visual=False)
    k=4
    times=[5]
    Tt=computeTt(G,times)
    dat,P=maxMS(k,times,G,gamma=0)
    p_alg=P[:,0]
    p_alg=np.array(p_alg)
    group_of_interest=counts[np.argmax(p_alg)] #should identify the group we analyze most of the time - if not then re-run

    it=500 #>200
    metric=np.zeros(it)
    prevCore=np.array(n_largest_indices(p_alg,1))
    full_set = set(range(G.number_of_nodes()))
    core_sizes=[200,300,400,500]
    maxVal=-1
    bestCore=list(full_set)
    for i in range(it):
        print(i)
        if i==0:
            Core=np.array(n_largest_indices(p_alg,1))
        else:
            Core=np.array(adjacent_core_seeded(p_alg,G,prevCore))
        Periphery = list(full_set - set(Core))
        metric[i]=corePeripheryMetricFast(Core,Periphery,Tt,G,p_alg,5)
        if (i+1) in core_sizes:
            print(f"Core Size: {i+1}")
            visualize_core(Core,xdat,ydat,counts,group_of_interest)
        if metric[i]>maxVal:
            maxVal=metric[i]
            bestCore=Core
        prevCore=Core
    visualize_data(xdat,ydat,counts,group_of_interest)
    print(len(bestCore))
#################################################

#################################################
#Core Periphery Tests on Road Networks - Figure 5.4 and Figure 5.5
#Code to generate data
if False:
    import matplotlib as plt
    bestcoresizes=[0,0]
    indexset=[7,9]
    for j in range(len(indexset)):
        G = nx.read_graphml('modified_bay_graph.graphml')
        old_labels = list(G.nodes())
        new_labels = list(range(len(old_labels)))
        mapping = dict(zip(old_labels, new_labels))
        #relabel nodes
        G = nx.relabel_nodes(G, mapping)
        n=G.number_of_nodes()
        t=5
        scipy_sparse = sp.csr_matrix(nx.adjacency_matrix(G))
        A=tf.SparseTensor(
            indices=np.array(scipy_sparse.nonzero()).T,
            values=scipy_sparse.data,
            dense_shape=scipy_sparse.shape
        )
        degrees = np.ravel(nx.adjacency_matrix(G)@np.ones(n))
        inv_degrees=[1/x for x in degrees]
        T=sparse.diags(inv_degrees)@scipy_sparse
        n_edge=G.number_of_edges()
        statDist=[x/(2*n_edge) for x in degrees]
        Tt=T.copy()
        for step in range(t-1):
            Tt=Tt@T
        P=np.load("Modified_Bay_Optimal_k_12.npy")
        p_alg=np.array(P[:,indexset[j]])
        m=75000 #was 75000
        l=1000 #was 1000
        inc=25

        testMetric=np.zeros(m)
        prevCore=n_largest_indices(p_alg,1)
        nlarg=n_largest_indices(p_alg,m)
        full_set = set(range(n))
        k=3
        threshold = np.quantile(p_alg, 1 - 1/k)
        p_indices = np.where(p_alg >= threshold)[0]
        #periphery_set=set(p_indices)
        periphery_set=full_set
        temp=np.arange(0,m,inc)
        maxVal=-1
        bestCore=list(full_set)
        Core=[]
        for i in temp: #range(m)
            if i%100==0:
                print(i)
            if len(Core)==0:
                Core=np.array(n_largest_indices(p_alg,1))
            else:
                Core=np.array(nlarg[:i])
            Periphery = np.array(list(periphery_set - set(Core)))
            testMetric[i]=corePeripheryMetricFast(Core,Periphery,Tt,G,p_alg,t)
            if testMetric[i]>maxVal and len(Core)>=l:
                maxVal=testMetric[i]
                bestCore=Core
            prevCore=list(Core)
        np.save(f"Road_Network_Data_Trial_{j}.npy",testMetric)
        #old plot code went here
        print(f"Length of Best Core: {len(bestCore)}")
        np.save(f"Road_Network_Best_Core_Trial_{j}.npy",bestCore)
        bestcoresizes[j]=len(bestCore)
    print(bestcoresizes)

#Code to produce plots of core metric as a function of core size
if False:
    import matplotlib as plt
    m=75000
    indexset=[7,9]
    for i in range(2):
        testMetric=np.load(f"Road_Network_Data_Trial_{i}.npy")
        x_values=np.array(list(range(m)))
        plt.pyplot.figure(figsize=(10, 7))
        plt.pyplot.plot(x_values[np.array(testMetric>0)]/1000, testMetric[np.array(testMetric>0)], '.')
        plt.pyplot.xlabel('Core Size (thousand)', fontsize=23)
        plt.pyplot.ylabel('Value', fontsize=23)
        plt.pyplot.tick_params(axis='both', labelsize=20)
        ax = plt.pyplot.gca()
        ax.yaxis.offsetText.set_fontsize(18)
        plt.pyplot.title("Core-Periphery Metric by Core Size", fontsize=24)
        plt.pyplot.show()
        plt.pyplot.savefig(f"Distribution_{indexset[i]}_k12_nLarg_Core_Periphery_Metric_Modified_Bay.pdf")

#Code to produce visualization of optimal core within hard partition
if False:
    import matplotlib as plt
    P=np.load("Modified_Bay_Optimal_k_12.npy")
    for i in range(2):
        bestCore=np.load(f"Road_Network_Best_Core_Trial_{i}.npy")
        nodes = pd.read_csv('modified_nodes_bay.csv')
        ydat=nodes.iloc[:,2]
        xdat=nodes.iloc[:,3]
        ind=nodes.index
        clustering=np.argmax(P,axis=1)
        color_map = {
        0: "red",        # Red
        1: "blue",       # Blue
        2: "green",      # Green
        3: "orange",     # Orange
        4: "purple",     # Purple
        5: "brown",      # Brown
        6: "pink",       # Pink
        7: "gray",       # Gray
        8: "cyan",       # Cyan
        9: "olive",      # Olive
        10: "gold",      # Gold (distinct yellow)
        11: "teal",
        12: "black"}  
        for c in bestCore:
            clustering[c]=12
        #make plots
        fig,ax=plt.pyplot.subplots(figsize=(8, 6))
        plt.pyplot.scatter(xdat, ydat, c=[color_map[i] for i in clustering], s=2, linestyle='None', marker='o', alpha=0.3)
        plt.pyplot.scatter(xdat[clustering==12], ydat[clustering==12], color='black',s=2,linestyle='None',marker='o')
        plt.pyplot.title('Plot of Road Network with Optimal Core', fontsize=30)
        ax.grid(True,alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.pyplot.show()
        plt.pyplot.savefig(f"Distribution_{[i]}_k12_nLarg_Core_Periphery_Modified_Bay.pdf")

#################################################

#################################################
#Appendix Figures
#Bay-Area Hard Partitions for Higher Values of k - Figure D.1
if False:
    import matplotlib as plt
    k_vals=[12,20]
    for k in k_vals:
        nodes = pd.read_csv('modified_nodes_bay.csv')
        P=np.load(f"Modified_Bay_Optimal_k_{k}.npy")
        ydat=nodes.iloc[:,2]
        xdat=nodes.iloc[:,3]
        ind=nodes.index
        clustering=np.argmax(P,axis=1)
        color_map = {
        0: "red",        # Red
        1: "blue",       # Blue
        2: "green",      # Green
        3: "orange",     # Orange
        4: "purple",     # Purple
        5: "brown",      # Brown
        6: "pink",       # Pink
        7: "gray",       # Gray
        8: "cyan",       # Cyan
        9: "olive",      # Olive
        10: "gold",      # Gold (distinct yellow)
        11: "teal"}       # Teal (distinct blue-green)
        color_map2 = {
        0: "red",        # Red
        1: "blue",       # Blue
        2: "green",      # Green
        3: "orange",     # Orange
        4: "purple",     # Purple
        5: "brown",      # Brown
        6: "pink",       # Pink
        7: "gray",       # Gray
        8: "cyan",       # Cyan
        9: "olive",      # Olive
        10: "gold",      # Gold (distinct yellow)
        11: "teal",      # Teal (distinct blue-green)
        12: "magenta",   # Magenta
        13: "lime",      # Lime
        14: "navy",      # Navy
        15: "maroon",    # Maroon
        16: "aqua",      # Aqua
        17: "coral",     # Coral
        18: "orchid",    # Orchid
        19: "salmon"}     # Salmon
        #make plots
        fig,ax=plt.pyplot.subplots(figsize=(8, 6))
        if k==20:
            cols=[color_map2[i] for i in clustering]
        else:
            cols=[color_map[i] for i in clustering]
        plt.pyplot.scatter(xdat, ydat, c=cols, s=2, marker='o')
        plt.pyplot.title('Bay Area Road Network Hard Partition',fontsize=20)
        ax.grid(True,alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.pyplot.savefig(f"Hard_Partition_k{k}_Modified_Bay.pdf")
        plt.pyplot.show()

#Eigengap Table - Table D.1
if False:
    import matplotlib as plt
    Vk=np.load("Modified_Bay_Vk25_data.npy")
    G = nx.read_graphml('modified_bay_graph.graphml')
    scipy_sparse = sp.csr_matrix(nx.adjacency_matrix(G))
    A=tf.SparseTensor(
    indices=np.array(scipy_sparse.nonzero()).T,
    values=scipy_sparse.data,
    dense_shape=scipy_sparse.shape
    )
    n=G.number_of_nodes()
    degrees = np.ravel(nx.adjacency_matrix(G)@np.ones(n))
    I = sparse.diags(np.ones(A.shape[0]))
    inv_sqrt_degrees=[1/np.sqrt(x) for x in degrees]
    D_inv_sqrt=sparse.diags(inv_sqrt_degrees)
    temp=D_inv_sqrt @ csr_matrix(nx.adjacency_matrix(G)) @ D_inv_sqrt
    LN = I - D_inv_sqrt @ csr_matrix(nx.adjacency_matrix(G)) @ D_inv_sqrt
    eigenvals=np.zeros(25)
    for i in range(25):
        evec=Vk[:,i]
        eigenvals[i]=np.linalg.norm(LN@evec)/np.linalg.norm(evec)
    eigengaps=np.zeros(24)
    for i in range(24):
        eigengaps[i]=eigenvals[i+1]-eigenvals[i]
    #normalize the eigengaps
    norm_eigengaps=(eigengaps-np.mean(eigengaps))/np.std(eigengaps)
    print(norm_eigengaps)
    for i in range(len(eigengaps)):
        print(f"k={i+1}:    {norm_eigengaps[i]}")

    plt.pyplot.figure(figsize=(8, 6))
    plt.pyplot.scatter(np.arange(1,25,1), eigengaps, s=10, marker='o')
    plt.pyplot.title('Plot of Eigengaps')
    plt.pyplot.xlabel('k')
    plt.pyplot.ylabel('mu_{k+1}-mu_k')
    plt.pyplot.grid(True)
    plt.pyplot.savefig("Eigengaps.png")
    plt.pyplot.show()
#################################################