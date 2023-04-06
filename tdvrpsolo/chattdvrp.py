import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import numpy as np
import itertools
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
import pytorch_lightning as pl

class TemporalGraphAttentionNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr, clip_epsilon):
        super(TemporalGraphAttentionNetwork, self).__init__()
        self.save_hyperparameters()

        self.gat1 = GATConv(input_size, hidden_size)
        self.gat2 = GATConv(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return Categorical(logits=self.policy_head(x)), self.value_head(x)

    def training_step(self, batch, batch_idx):
        policy, value = self(batch)
        action = policy.sample()
        log_prob = policy.log_prob(action)

        # Calculate rewards based on travel time
        rewards = -batch.travel_time.gather(1, action.unsqueeze(1)).squeeze()

        # Calculate advantage for each action
        advantages = rewards - value.squeeze()

        if not hasattr(batch, 'old_log_probs'):
            batch.old_log_probs = log_prob.detach()
        else:
            batch.old_log_probs = batch.old_log_probs.detach()

        # PPO objective function
        ratio = torch.exp(log_prob - batch.old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.hparams.clip_epsilon, 1 + self.hparams.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + value_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

def create_data(num_nodes,seed=1234):
    # Generate random positions for nodes
    np.random.seed = seed
    positions = np.random.rand(num_nodes)
    
    # Compute distances (travel times) between nodes
    travel_times = np.sqrt(np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1))
    
    # Create edge list
    edges = list(itertools.combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create travel time tensor
    travel_times_tensor = torch.tensor(travel_times, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=torch.tensor(positions, dtype=torch.float), edge_index=edge_index, travel_time=travel_times_tensor)

    return data

def main():
    # Create a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        log_every_n_steps= 5
    )

    # Define hyperparameters
    hparams = {
        "lr": 1e-3,
        "clip_epsilon": 0.2,
        "input_size": 2,
        "hidden_size": 64,
        "output_size": 1,
    }

    # Create and train the model
    model = TemporalGraphAttentionNetwork(**hparams)

    # Create data for different numbers of nodes
    data_10_nodes = create_data(10,1234)
    
    # Combine the data into a single DataLoader
    dataset = [data_10_nodes]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, follow_batch=[])

    # Train the model with the DataLoader
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()