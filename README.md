# PlanIT: Planning and Instantiating Indoor Scenes with Relation Graph and Spatial Prior Networks
PyTorch code for our SIGGRAPH Paper [PlanIT: Planning and Instantiating Indoor Scenes with Relation Graph and Spatial Prior Networks](https://kwang-ether.github.io/pdf/planit.pdf)

The code is provided as is, we could not run any further tests to ensure that it is working since we no longer have access to the dataset we used in the paper. We cannot provide anything derived from the dataset, including the pre-trained models, as well as several metadata files that our code rely on. We will update this in the event that this situation changes in the future. In the meanwhile, we are willing to answer questions for those we wish to adapt this to a different dataset.

The data process pipeline is similar to our work from SIGGRAPH 2018, please refer to [the repo for that paper](https://github.com/brownvc/deep-synth) for details.

The image-based instantiation pipeline is mostly adapted from our work from CVPR 2019, please refer to [the repo for that paper](https://github.com/brownvc/fast-synth) for details.

`/scene-synth/data/graph.py` contains the code used for graph extraction and data-driven graph pruning.

`/scene-synth/models/graph.py` contains the code for the graph generative model.

`/scene-synth/loc.py` contains the conditional location sampler.

`/scene-synth/graph-synth.py` contains the main pipeline for scene synthesis.

Apologies again for being unable to provide more details for running the code.
