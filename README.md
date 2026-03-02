# DHEG-PPIS: Predicting protein-protein interaction sites based on Dynamic perception mechanism within a Hierarchical E(n)-equivariant Graph architecture


---

### 🚀 Features

Accurate prediction of protein-protein interaction sites (PPIS) is crucial to understanding biological processes, elucidating disease mechanisms, and accelerating drug discovery. Although graph neural network (GNN) methods have shown potential in this field, but existing methods are limited by the static integrate multi-group features and insufficient perception of hierarchical 3D spatial geometric information, leading to insufficient predictive ability of orphan sites. To address these issues, this paper proposes a \textbf{D}perception mechanism within a \textbf{H}ierarchical \textbf{E}(n)-equivariant \textbf{G}raph architecture (DHEG). DHEG introduces a dynamic feature importance perception mechanism that adaptively perceives the contextual inter-dependencies of features and assigns weights to feature groups based on their relevance to the interaction relationship. And a hierarchical gated architecture based on E(n)-equivariant graph neural networks that effectively captures protein 3D spatial structures while mitigating over-smoothing problems. The results show that DHEG achieves improvements in 11 of 13 key metrics, with an enhancement 8\% in MCC, indicating that DHEG not only predicts more interaction sites but also does so with greater reliability. Furthermore, case studies and visualization analyzes show that DHEG aligns better with the biological mechanism and has excellent predictive capabilities for both orphan sites and continuous regions, demonstrating interpretability and application potential.

---
##  Introduction
DHEG is a novel framework for protein-protein interaction site prediction using Dynamic perception mechanism within a Hierarchical E(n)-equivariant Graph architecture. DHEG introduces a dynamic feature perception mechanism that adaptively weights multiple protein features, based on their contextual relevance, overcoming the rigidity of static feature integration. Furthermore, leveraging the inherent geometric symmetry of 3D protein structures, DHEG incorporates an EGNN to construct a hierarchical gated architecture. This architecture utilizes graph attention pooling to establish hierarchical representations from graphs to subgraphs and enables cross-level information fusion through a gating mechanism, effectively capturing multi-scale spatial geometric features while mitigating over-smoothing issues. Comparative experiments demonstrate that DHEG significantly outperforms existing state-of-the-art (SOTA) methods, and its predicted interaction sites show strong agreement with experimentally determined interfaces. 
 

## System requirement
torch>=2.1.0
torchvision>=0.16.0
dgl>=1.1.2
numpy>=1.25.0
pandas>=2.0.3
scikit-learn>=1.3.2
matplotlib>=3.8.0
einops>=0.8.0
tqdm>=4.66.0
pickle5>=0.0.12
Software and database requirement
To run the DHEG, you need to install the following three software and download the corresponding databases:
BLAST+ [https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/] and UniRef90 [https://www.uniprot.org/downloads], run PSSM.py
HH-suite [https://github.com/soedinglab/hh-suite] and Uniclust30 [https://uniclust.mmseqs.com/], run HMM.py
DSSP [https://github.com/cmbi/dssp], run DSSP.py
Run DHEG
Python train.py
Python test.py
Notice: Replace the model path 


## Others
If you have any questions, please do not hesitate to connect us [bencaocs@gmail.com or xueleecs@gmail.com].


