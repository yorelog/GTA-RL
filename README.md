# Dynamic Graph Combinatorial Optimization with Multi-Attention Deep Reinforcement Learning

Spatio-Temporal architecture to solve dynamic combinatorial optimization and this code base build on top of the following code (https://github.com/wouterkool/attention-learn-to-route). 

Currently, dynamic versions of Travelling Salesmen Problem (TSP) and Vehicle Routing Problem (VRP) are supported. 

## Installation

Install following libraries

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

Or use [env.yml](env.yml) for the complete list of libraries. You can use Anaconda for the installation.

## Generate Data

Use the following script to generate test graph instances.

E.g: To generate 100 graphs with 20 nodes for dynamic TSP
```bash
python generate_data.py 
  --problem dynamic_tsp 
  --name <dataset_name> 
  --seed 4321 
  --graph_size 20 
  --dataset_size 100
```

For dynamic VRP use --problem dynamic_cvrp

## Train GTA-RL

Use the following script to start training GTA-RL for dynamic TSP with 

```bash
python run.py
  --problem dynamic_tsp
  --model st_attention
  --graph_size 20 
  --baseline rollout 
  --run_name 'dynamic_tsp' 
  --val_dataset <location of the generated validation dataset>
```

If the --val_dataset is not provided, the validation dataset will be automatically generated.

To enable GTA-RL in real-time mode use  "--use_single_time"

Refer to [options.py](options.py) for the complete list of parameters

## Pretrained models

The pretrained models are available under [pretrained](pretrained) folder. 
E.g. [pretrained/dynamic_tsp_20](pretrained/dynamic_tsp_20) contains the trained model for dynamic tsp with 20 nodes

## Testing

Use the following script to evaluate the trained model.

```bash
python eval.py <location of the generated dataset> 
  --model pretrained/dynamic_tsp_20/GTA_RL
  --decode_strategy <decode stratergy> 
  --eval_batch_size 1
```

Possible options for --decode_strategy are "greedy" and "bs" (for beam search). Use --beam_width <int> with "bs" option.

## Visualization
Use the following script to visualize the solution.
  
  
```bash
python eval.py data/tsp/tsp20_test_seed1234.pkl 
  --model pretrained/dynamic_tsp_20 
  --decode_strategy <decode stratergy> 
  --eval_batch_size 1
  --plot
  --plot_index <index of the batch to plot>
```
For now, only the dynamic TSP problem works with plotting. Use --use_gurobi to plot the solution with gurobi to the same plot_index
  
## Acknowledgements
We thank attention learning to route [https://github.com/wouterkool/attention-learn-to-route] for an easily extendable codebase. 
