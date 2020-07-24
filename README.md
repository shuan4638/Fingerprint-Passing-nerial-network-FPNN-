# FP2GRAPH
Predicting chemical property using descriptors from Morgan Fingerprints to graph convolutional neural network (GCN).

HIV dataset was used to demo classification task and ESOL dataset was used to demo regression task.

## AtomConvolution

In addition to graph convolution, this code also performed "atom convolution" to more closely learn the chemical behavior.
<img src="https://github.com/shuan4638/FP2GRAPH/blob/master/Atomconv.jpg">
## Installation

All the chemical decription were done by rdkit. Please use conda to install rdkit.

```bash
conda install -c conda-forge rdkit
```

## Author
This code was written by Shuan Chen (PhD candidate of KAIST CBE) in 2019 for chemical GCN practice.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

This work was inspired by 2 papers:

1. "Convolutional networks on graphs for learning molecular fingerprints", Neural information processing systems, pp. 2224–2232, 2015

2. "FP2VEC: A new molecular featurizer for learning molecular properties", Bioinformatics, 35 (23), pp. 4979-4985, 2019

