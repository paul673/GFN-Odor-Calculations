import torch
from gflownet import GFNTask, ObjectProperties, LogScalar
from typing import Dict, List, Tuple
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.config import init_empty, Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext


dev = torch.device('cpu')


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

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        # This method computes object properties, these can be anything we want
        # and aren't the reward yet.
        # We return an (n, 1) scalar, as well as a (n,) tensor of bools indicating
        # whether the objects are valid. In our case, they all are, but the task
        # may choose to invalidate objects it doesn't want.
        #rs = torch.tensor([m.GetRingInfo().NumRings() for m in mols]).float()

        # TODO : calculate the proper reward property Maximises flour at the moment.
        #rs = torch.tensor([fluor_counter(m) for m in mols]).float()
        rs = torch.tensor([calculate_similarity(m) for m in mols]).float()


        return ObjectProperties(rs.reshape((-1, 1))), torch.ones(len(mols)).bool()

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        # This method transforms the object properties we computed above into a
        # LogScalar, more precisely a log-reward, which will be passed on to the
        # learning algorithm.
        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-30).log()
        return LogScalar(scalar_logreward.flatten())
    

class SensesTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        # Here we choose some specific parameters, in particular, we don't want
        # molecules of more than 10 atoms, we we set
        cfg.algo.max_nodes = 7

        # This creates a lagged sampling model, see https://arxiv.org/abs/2310.19685
        cfg.algo.sampling_tau = 0.9

        # It is possible that our GFN generates impossible molecules. This will be
        # their logreward:
        cfg.algo.illegal_action_logreward = -75
        # Disable random actions
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0

    def setup_task(self):
        # The task we created above
        self.task = SensesTask()

    def setup_env_context(self):
        # The per-atom generation context
        self.ctx = MolBuildingEnvContext(
            max_nodes=self.cfg.algo.max_nodes,  # Limit the number of atoms
            num_cond_dim=1,  # As per sample_conditional_information, this will be torch.ones((n, 1))
        )




config = init_empty(Config())
config.print_every = 1
config.log_dir = "./log_dir_smiles"
config.device = dev
config.num_training_steps = 10
config.num_workers = 0
config.num_validation_gen_steps = 1
config.overwrite_existing_exp=True

trial = SensesTrainer(config, print_config=False)
trial.run()


#pip install -e "D:\Dokumenter\Skole\Prosjektoppgave\gflownet"  --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
# pip
# 
# pip install numpy<2     