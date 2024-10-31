# Import libaries
import numpy as np
import torch

from typing import Dict, List, Tuple

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.online_trainer import StandardOnlineTrainer
from torch import Tensor
from rdkit.Chem.rdchem import Mol as RDMol

from gflownet.config import Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext


from rdkit import Chem

from pom_models.functions import fragance_propabilities_from_smiles







class SensesTask(GFNTask):
    """A task for the senses model."""

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        # This method exists to initiate trajectories that may depend on different
        # conditional information. For example, we could tell the model to generate
        # molecules with a logP between 3 and 4, and penalize it (in the
        # cond_info_to_logreward method) if it doesn't.

        # Because we don't want to make the generation conditional on anything, we
        # provide a constant "encoding" vector. We also don't care about a reward
        # temperature, so provide a constant beta = 1
        return {"beta": torch.ones(n), "encoding": torch.ones(n, 1)}


    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        # This method transforms the object properties we computed above into a
        # LogScalar, more precisely a log-reward, which will be passed on to the
        # learning algorithm.
        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-30).log()
        return LogScalar(scalar_logreward.flatten())

class MoleculeTask(SensesTask):
    """
    Task to generate molecules with fragance nodes compareable to a specified molecule.
    """

    def __init__(self, smiles,num_important_frag_notes=5, weight=10):
        super().__init__()
        self.smiles = smiles
        self.num_important_frag_notes = 1 if num_important_frag_notes < 1 else num_important_frag_notes
        self.mol_prob = fragance_propabilities_from_smiles(self.smiles)[0]
        
        # Edge case multible notes with same probability inform that the number of fragance notes have been increased
        max_probs = sorted(self.mol_prob, reverse=True)[:num_important_frag_notes]  
        
        additional_notes = list(self.mol_prob).count(max_probs[-1]) - list(max_probs).count(max_probs[-1])
        if additional_notes>0:
            print(f"Fragrance notes with equal probablility discovered. Increase number of important notes to {additional_notes+num_important_frag_notes}")
        
        self.mask = list(map(lambda x:  1 if x>max_probs[-1] else  0, self.mol_prob))
        self.weight = weight



    def reward_function(self,mol):

        # Skip model evaluation for molecules with one atom to prevent the pom model 
        # from crashing. Set the reward for this case to 1. 
        atoms=mol.GetAtoms()
        if len(atoms) <= 1:
            return 1
        
        # Evaluate the molecules probabilities for different fragance notes 
        smiles = Chem.MolToSmiles(mol)
        probabilities = fragance_propabilities_from_smiles(smiles)

        # Reward molecules with a high probability for the five most important 
        # fragrance notes for vanilla. The mask is multiplied by 10 to increase 
        # the weight compared to the reward for molecules with just one atom
        reward_array = np.array(self.mask) * self.weight
        return  float(sum((probabilities * reward_array)[0]))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        # This method computes object properties, these can be anything we want
        # and aren't the reward yet.
        # We return an (n, 1) scalar, as well as a (n,) tensor of bools indicating
        # whether the objects are valid. In our case, they all are, but the task
        # may choose to invalidate objects it doesn't want.

        rs = torch.tensor([self.reward_function(m) for m in mols]).float()
        return ObjectProperties(rs.reshape((-1, 1))), torch.ones(len(mols)).bool()
    


    # This is the basic GFN trainer that we'll reuse for our purposes


class MoleculeTrainer(StandardOnlineTrainer):
    def __init__(self, config, smiles, print_config=True, num_important_frag_notes=5, weight=10):
        self.smiles = smiles
        self.num_important_frag_notes=num_important_frag_notes
        self.weight = weight
        super().__init__(config, print_config)
        


    def set_default_hps(self, cfg: Config):
        # Here we choose some specific parameters, in particular, we don't want
        # molecules of more than 7 atoms, we we set
        cfg.algo.max_nodes = 7

        # This creates a lagged sampling model, see https://arxiv.org/abs/2310.19685
        cfg.algo.sampling_tau = 0.9

        # It is possible that our GFN generates impossible molecules. This will be
        # their logreward:
        cfg.algo.illegal_action_logreward = -75
        # Disable random actions
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0


        cfg.algo.num_from_policy = 64
        #cfg.algo.num_from_dataset=5

    def setup_task(self):
        # The task we created above
        self.task = MoleculeTask(self.smiles, num_important_frag_notes=self.num_important_frag_notes, weight=self.weight)

    def setup_env_context(self):
        # The per-atom generation context
        self.ctx = MolBuildingEnvContext(
            ["C","N","O"],
            max_nodes=self.cfg.algo.max_nodes,  # Limit the number of atoms
            num_cond_dim=1,  # As per sample_conditional_information, this will be torch.ones((n, 1))
            charges=[0],  # disable charge
            chiral_types=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED],  # disable chirality
            num_rw_feat=0, #how many features are associated with each node during the random walk process. 
            expl_H_range=[0],
        )



