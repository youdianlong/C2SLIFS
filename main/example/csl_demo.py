from main.common import GraphDAG
from main.metrics import MetricsDAG
from main.datasets import DAG, IIDSimulation
from main.algorithms import CSL

weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10, weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='nonlinear', sem_type='mlp')
true_dag, X = dataset.B, dataset.X
'''
from cdt.data import load_dataset
s_data, s_graph = load_dataset('sachs')
X = s_data[0:853]
true_dag = nx.to_numpy_array(s_graph)
'''
csl = CSL(model_type='nn',iter_step=100,rho_thresh=1e20,init_rho=1e-5,
          rho_multiply=10,graph_thresh=5.0001e-1,l1_graph_penalty=2e-3)
csl.learn(X, pns_mask=true_dag)

# plot est_dag and true_dag
GraphDAG(csl.causal_matrix, true_dag)

# calculate accuracy
met = MetricsDAG(csl.causal_matrix, true_dag)
print(met.metrics)