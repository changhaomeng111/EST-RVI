# **ST-AugGCN: Spatio-Temporal Augmented Graph Convolutional Network for Traffic Risk Prediction**
 **ST-AugGCN** (Spatio-Temporal Augmented Graph Convolutional Network), a novel framework for traffic risk prediction that effectively models complex spatio-temporal dependencies and external events.

 ## 📖 Overview
 
ST-AugGCN is an advanced deep learning framework designed for accurate traffic flow prediction by capturing:

- **Spatial Dependencies**: Multi-scale graph convolutional networks to model road network topology
- **Temporal Dynamics**: Gated recurrent units with attention mechanisms for temporal patterns
- **External Events**: Quadruple-based event encoding for weather, queries, and incidents
- **Cross-Modal Interactions**: Contrastive learning for robust feature representation

## 🎯 Custom Events and Graph Structures
### Replacing Graph Structure
(1) Initialize constructor
  ```bash
  constructor = MotifGraphConstructor(
  
      speed_matrix_path=config['speed_matrix_path'],
      
      adj_matrix_path=config['adj_matrix_path'],
      
      k_neighbors=config['k_neighbors'],
      
      dtw_threshold=config['dtw_threshold'])
  ```
(2) Construct motif graph

    motif_graph = constructor.construct_motif_graph()
   
(3) Analyze motif distribution

    motif_distribution = constructor.analyze_motifs()
   
(4) Save motifs

    constructor.save_graph(
      graph_path=config['output_graph_path'],
      adj_matrix_path=config['output_adj_path'])

### Adding New Event Types
Define new event types in data processor

new_events = ['accident', 'construction', 'special_event']

The model will automatically learn to incorporate new event types
through the quadruple encoding mechanism

### DataSets
The aforementioned documents already contain the New York City dataset.

The Q-traffic dataset get from (https://github.com/JingqingZ/BaiduTraffic).

The online crowd queries which are derived from users’ navigation or search queries can characterize the potential events at specific locations.

The traffic conditions and crowd queries for these road segments are sampled every 5 or 15 minutes.

### Requirements

- Python 3.6+
- TensorFlow 1.15.0
- NumPy 1.19.5
- Pandas 1.1.5
- SciPy 1.5.4
- scikit-learn 0.24.2

### Quick Install
```bash
# Clone the repository
git clone https://github.com/cm-m1/ST-AugGCN.git
cd ST-AugGCN

# Install dependencies
pip install -r requirements.txt
