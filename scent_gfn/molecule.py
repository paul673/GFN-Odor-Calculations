# Import libaries
import numpy as np
import torch
from numpy.linalg import norm

from typing import Dict, List, Tuple

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.online_trainer import StandardOnlineTrainer
from torch import Tensor
from rdkit.Chem.rdchem import Mol as RDMol

from gflownet.config import Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext


from rdkit import Chem,DataStructs

from pom_models.functions import fragance_propabilities_from_smiles


from rdkit.Chem.AtomPairs.Utils import CosineSimilarity

from scent_gfn.fragments import FRAGMENTS, FRAGMENTS_OPENPOM_DATASET, FRAGMENTS_OPENPOM_VANILLA





from torch.utils.data import Dataset
import pandas as pd
import random

class VanillaDataset(Dataset):
    def __init__(self, train=True, data_frame=None, ratio=1, split_seed=142857):
        self.split_seed=split_seed
        # Create a separate random generator for this instance
        self.random_gen = random.Random(self.split_seed)

        self.ratio = ratio
        if type(data_frame)==pd.DataFrame:
            self.df = data_frame
        else:
            df = pd.read_csv('../data/data.csv')
            self.df = df
            #self.df = df.loc[df['vanilla'] == 1]

        idcs = np.array(self.df.index)
        self.random_gen.shuffle(idcs)

        
        if train:
            self.idcs = idcs[:int(np.floor(ratio*len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio*len(self.df))):]

        self.obj_to_graph = lambda x: x


    def setup(self, task,ctx):
        self.obj_to_graph=ctx.obj_to_graph


    
    def __getitem__(self, idx):
        return (
            self.obj_to_graph(Chem.MolFromSmiles(self.df["nonStereoSMILES"][self.idcs[idx]])),
            torch.tensor([fragance_propabilities_from_smiles(self.df["nonStereoSMILES"][self.idcs[idx]])[0][-9]]).float()#torch.tensor(self.df.loc[self.idcs[idx]][2:]).float()
        )
    
    def __len__(self):
        return len(self.idcs)
        




class SensesTask(GFNTask):
    """A task for the senses model."""




    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        # This method transforms the object properties we computed above into a
        # LogScalar, more precisely a log-reward, which will be passed on to the
        # learning algorithm.

        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-30).log()
        assert "beta" in cond_info.keys()
        beta = cond_info["beta"]
        scalar_logreward = scalar_logreward*beta
        return LogScalar(scalar_logreward.flatten())
    

    def is_chemically_realistic(self, mol):
        for atom in mol.GetAtoms():
            # Retrieve the atomic number and degree (number of bonds)
            atomic_num = atom.GetAtomicNum()
            valence = atom.GetExplicitValence() + atom.GetImplicitValence()
            # Add custom checks for phosphorus and other elements
            if atomic_num == 15:  # Atomic number for Phosphorus
                if valence > 5:  # P cannot exceed pentavalency
                    return False
        return True



    def has_unpaired_electrons(self, mol):
        """
        Check if a molecule has unpaired electrons (radicals).
        Args:
            mol (RDKit Mol): Molecule to check.
        Returns:
            bool: True if there are unpaired electrons, False otherwise.
        """
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return True
        return False
    
    def high_carbon_content(self, mol, threshold=0.6):
        """
        Check if a molecule has a carbon content below a certain threshold.
        Args:
            mol (RDKit Mol): Molecule to check.
            threshold: minimum carbon content threshold. Default 0.6
        Returns:
            bool: True if carbon content above threshold, False otherwise.
        """
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return False
        atoms = mol.GetAtoms()
        num_atoms = mol.GetNumAtoms()
        # Initialize the counter for carbon atoms
        carbon_count = 0

        # Iterate over all atoms in the molecule and check if the atom is carbon
        for atom in atoms:
            if atom.GetSymbol() == 'C':  # 'C' is the symbol for carbon
                carbon_count += 1
        carbon_content = carbon_count/num_atoms

        return carbon_content >= threshold
    
    def is_large_molecule(self,mol,max_num_atoms=15):
        num_of_atoms = mol.GetNumAtoms()
        if num_of_atoms <= max_num_atoms:
            return False
        return True


        

class MoleculeTask(SensesTask):
    """
    Task to generate molecules with fragance nodes compareable to a specified molecule.
    """

    def __init__(self, data_tuple:tuple[str,str,str,list,float]): # (reward func type , similarity measure)
        super().__init__()
        sim_func_dict = {
            "openpom":{
                "func": self.reward_function_openpom,
                "cosine": self.cosine_similarity,
                "tanimoto": self.cosine_similarity,
            },
            "structure":{
                "func": self.reward_function_structure,
                "cosine": CosineSimilarity,
                "tanimoto": CosineSimilarity,
            }
        }
        reward_function_type = data_tuple[0]
        similarity_measure = data_tuple[1]
        penalty = data_tuple[2]
        data = data_tuple[3]
        self.penalty = penalty
        exp_penalty = penalty[0] == "exponential"
        rew_func = sim_func_dict[reward_function_type]["func"]
        sim_func = sim_func_dict[reward_function_type][similarity_measure]
        self.beta = data_tuple[4]


        
        if type(data[0])== str:
            print("SMILES input data detected ...")
            self.init_mol_data(data)
        else:
            print("OpenPOM input data detected ...")
            self.init_pom_data(data)

        print(f"Reward function: {rew_func.__name__} | Similarity measure: {sim_func.__name__} | exp_penalty: {exp_penalty} | max_mol={penalty[1]}")
        self.selected_reward_func = lambda mol: rew_func(mol, sim_func, penalty=exp_penalty, max_num_atoms=penalty[1] if exp_penalty else 0)

        """if reward_function_type=="openpom": 
            if similarity_measure=="cosine":
                print("Reward function: OpenPOM with cosine similarity")
                self.selected_reward_func = lambda mol: self.reward_function_openpom(mol, self.cosine_similarity, penalty=exp_penalty)
            else:
                print("Reward function: OpenPOM with tanimoto similarity")
                self.selected_reward_func = lambda mol: self.reward_function_openpom(mol, self.cosine_similarity, penalty=exp_penalty)
        else:
            if similarity_measure=="cosine":
                print("Reward function: Structure with cosine similarity")
                self.selected_reward_func = lambda mol: self.reward_function_structure(mol, CosineSimilarity, penalty=exp_penalty)
            else:
                print("Reward function: Structure with tanimoto similarity")
                self.selected_reward_func = lambda mol: self.reward_function_structure(mol, CosineSimilarity, penalty=exp_penalty)"""


        #print(dataset)
        #self.training_data_smiles = []#[dataset.df["nonStereoSMILES"]]
        #self.idcs = dataset.idcs
        #for i in self.idcs:
           # self.training_data_smiles.append(dataset.df["nonStereoSMILES"][i])
        #self.target_smiles = target_smiles
        #self.num_target_mols = 1 if len(self.training_data_smiles) == 0 else len(self.training_data_smiles)
        #self.num_important_frag_notes = 1 if num_important_frag_notes < 1 else num_important_frag_notes
        #self.mol_prob = fragance_propabilities_from_smiles(self.smiles)[0]
        
        # Edge case multible notes with same probability inform that the number of fragance notes have been increased
        #max_probs = sorted(self.mol_prob, reverse=True)[:num_important_frag_notes]  
        
        #additional_notes = list(self.mol_prob).count(max_probs[-1]) - list(max_probs).count(max_probs[-1])
        #if additional_notes>0:
            #print(f"Fragrance notes with equal probablility discovered. Increase number of important notes to {additional_notes+num_important_frag_notes}")
        
        #self.mask = list(map(lambda x:  1 if x>max_probs[-1] else  0, self.mol_prob))
        #self.weight = weight
        #self.target_mols = [Chem.MolFromSmiles(smile) for smile in self.training_data_smiles]
        
        #self.target_probs = []#[fragance_propabilities_from_smiles(smile)[0] for smile in self.training_data_smiles]
        #for smile in self.training_data_smiles:
            #self.target_probs.append(fragance_propabilities_from_smiles(smile)[0] )
        #self.fpgen = Chem.AllChem.GetRDKitFPGenerator() # fingerprint generator


    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        # This method exists to initiate trajectories that may depend on different
        # conditional information. For example, we could tell the model to generate
        # molecules with a logP between 3 and 4, and penalize it (in the
        # cond_info_to_logreward method) if it doesn't.

        # Because we don't want to make the generation conditional on anything, we
        # provide a constant "encoding" vector. We also don't care about a reward
        # temperature, so provide a constant beta = 1
        return {"beta": torch.ones(n)*self.beta, "encoding": torch.ones(n, 1)}
        #torch.tensor(np.linspace(32,1,100))
        #return {"beta": torch.tensor(np.linspace(32,1,n)), "encoding": torch.ones(n, 1)}

    def init_mol_data(self, data):
        #print("Reading SMILES string vector training data ...")
        self.training_data_smiles = data
        self.num_target_mols = len(self.training_data_smiles)
        self.target_mols = [Chem.MolFromSmiles(smile) for smile in self.training_data_smiles]
        self.target_probs = []
        self.target_fingerprints = [Chem.RDKFingerprint(target_mol) for target_mol in self.target_mols]
        for smile in self.training_data_smiles:
            self.target_probs.append(fragance_propabilities_from_smiles(smile)[0] )
        return
    
    def init_pom_data(self,data):
        #print("Reading OpenPOM vector training data ...")
        self.target_probs = data
        self.target_mols = None
        self.training_data_smiles = None
        self.num_target_mols = len(data)
        return
    
    def reward_function(self,mol):
        return self.selected_reward_func(mol)


    def reward_function_legacy(self,mol):

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
        reward = float(sum((probabilities * reward_array)[0]))
        return reward
    
    def reward_function_structure(self,mol, similarity_func, penalty=False,max_num_atoms=15):
        """
        Reward function using cosine similarities for comparing molecules.
        """

        # Skip model evaluation for molecules with one atom to prevent the pom model 
        # from crashing. Set the reward for this case to 1. 
        #atoms=mol.GetAtoms()
        #if len(atoms) <= 1:
            #return 0
        
        # Evaluate the molecules probabilities for different fragance notes 
        #smiles = Chem.MolToSmiles(mol)
        #probabilities = fragance_propabilities_from_smiles(smiles)

        # Compare molecule to dataset with molecules with desired properties
        # and compute average similarity (between 0 and 1)
        #reward = 0
        #for target_mol in self.target_mols:
           # reward += CosineSimilarity(Chem.RDKFingerprint(mol),Chem.RDKFingerprint(target_mol))
        # reward = reward/self.num_target_mols
        reward = 0
        for target_fingerprint in self.target_fingerprints:
            new_reward = similarity_func(Chem.RDKFingerprint(mol),target_fingerprint)
            reward = new_reward if new_reward > reward else reward
        #reward = reward/self.num_target_mols


        # Reward molecules with a high probability for the five most important 
        # fragrance notes for vanilla. The mask is multiplied by 10 to increase 
        # the weight compared to the reward for molecules with just one atom
        #reward_array = np.array(self.mask) * self.weight
        #reward = float(sum((probabilities * reward_array)[0]))

        #Penalty for large molecules
        if penalty:
            reward = self.large_molecule_penalty(reward,mol, max_num_atoms=max_num_atoms)

        return reward
    
    
    
    def cosine_similarity(self, vec1,vec2):
        return np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    
    def reward_function_openpom(self,mol, similarity_func, penalty=False,max_num_atoms=15):
        """
        Reward function using cosine similarities for comparing fragrance note probabilities.
        """

        # Penalize invalid molecules or molecules with unpaired electrons
        #if not self.is_valid_molecule(mol):
           # return -1

        # Skip model evaluation for molecules with one atom to prevent the pom model 
        # from crashing. Set the reward for this case to 1. 
        atoms=mol.GetAtoms()
        if len(atoms) <= 1:
            return 0
        
        # Evaluate the molecules probabilities for different fragance notes 
        smiles = Chem.MolToSmiles(mol)
        probabilities = fragance_propabilities_from_smiles(smiles)[0]
        
        # Compare molecule to dataset with molecules with desired properties
        # and compute average similarity (between 0 and 1)
        #reward = 0
        #for target_probability in self.target_probs:
            #target_probabilities = fragance_propabilities_from_smiles(target_smile)[0]
            #reward += self.cosine_similarity(probabilities,target_probability)
        #reward = reward/self.num_target_mols
        reward = 0
        for target_probability in self.target_probs:
            #target_probabilities = fragance_propabilities_from_smiles(target_smile)[0]
            new_reward = similarity_func(probabilities,target_probability)
            reward = new_reward if new_reward > reward else reward
        #reward = reward/self.num_target_mols


        # Reward molecules with a high probability for the five most important 
        # fragrance notes for vanilla. The mask is multiplied by 10 to increase 
        # the weight compared to the reward for molecules with just one atom
        #reward_array = np.array(self.mask) * self.weight
        #reward = float(sum((probabilities * reward_array)[0]))

        #Penalty for large molecules
        if penalty:
            reward = self.large_molecule_penalty(reward,mol,max_num_atoms=max_num_atoms)

        return reward
    

    def is_valid_molecule(self, mol):
        try:
            # Standard RDKit sanitization
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            # Check for unpaired electrons
            if self.has_unpaired_electrons(mol):
                #print(f"unpaired electrons {Chem.MolToSmiles(mol)}")
                return False
            # Additional chemical realism checks
            if not self.is_chemically_realistic(mol):
                #print(f"Not chemical realistic {Chem.MolToSmiles(mol)}")
                return False
            # Ensure high enough carbon content
            if not self.high_carbon_content(mol):
                #print(f"low carbon content {Chem.MolToSmiles(mol)}")
                #print(Chem.MolToSmiles(mol))
                return False
            
            if self.penalty[0] == "hard":
                #print("large")
                if self.is_large_molecule(mol,max_num_atoms=self.penalty[1]):
                    return False


            return True
        
            
        except Exception:
            #print("exept")
            return False

    def large_molecule_penalty(self, reward, mol,max_num_atoms = 15):
        """
        Penalty for generating large molecules. The 75th percentile (3rd quartile) of the openpom dataset contains molecules with up to 15 atoms. 
        This function decreases the reward exponentially if the length of the molecule exceeds max_num_atoms. The 95th percentile reward at 20 atoms will 
        be decreased to approximatly 60% of its original value. 
        """
        num_of_atoms = mol.GetNumAtoms()
        if num_of_atoms <= max_num_atoms:
            return reward
        return reward * np.exp(-0.1*(num_of_atoms-max_num_atoms))


    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        # This method computes object properties, these can be anything we want
        # and aren't the reward yet.
        # We return an (n, 1) scalar, as well as a (n,) tensor of bools indicating
        # whether the objects are valid. In our case, they all are, but the task
        # may choose to invalidate objects it doesn't want.
        is_valid = torch.tensor([self.is_valid_molecule(m) for m in mols]).bool()
        #print([Chem.MolToSmiles(m) for m in mols])
        #print(is_valid)
        #print(torch.ones(len(mols)).bool(),torch.ones(len(mols)).bool().size())
        #print(valid_mols_mask,valid_mols_mask.size())
        #rs = torch.tensor([self.reward_function(m) for m in mols]).float()
        #return ObjectProperties(rs.reshape((-1, 1))), valid_mols_mask
            # Compute rewards only for valid molecules

        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid
        rewards = []
        for m, is_valid_obj in zip(mols, is_valid):
            if is_valid_obj:
                rewards.append(self.reward_function(m))
        rs = torch.tensor(rewards).float().reshape((-1, 1))
        
        assert len(rs) == is_valid.sum()

        return ObjectProperties(rs), is_valid
        # Convert rewards to a tensor
        #rs = torch.tensor(rewards).float()
        #print(rs.reshape((-1, 1)), rs.reshape((-1, 1)).size())
        #print(valid_mols_mask,valid_mols_mask.size())
        # Ensure rewards are reshaped to (n, 1)
        #return ObjectProperties(rs.reshape((-1, 1))), valid_mols_mask
    


    # This is the basic GFN trainer that we'll reuse for our purposes


class MoleculeTrainer(StandardOnlineTrainer):
    def __init__(self, config, data_tuple, fragments=None, print_config=True):
        self.data_tuple = data_tuple
        self.fragments = fragments if fragments else None
        #self.target_smiles = target_smiles
        #self.num_important_frag_notes=num_important_frag_notes
        #self.weight = weight
        super().__init__(config, print_config)
        


    def set_default_hps(self, cfg: Config):
        # Here we choose some specific parameters, in particular, we don't want
        # molecules of more than 7 atoms, we we set
        #cfg.algo.max_nodes = 20 # 95 quantil

        # This creates a lagged sampling model, see https://arxiv.org/abs/2310.19685
        cfg.algo.sampling_tau = 0.9

        # It is possible that our GFN generates impossible molecules. This will be
        # their logreward:
        
        # Disable random actions
        #cfg.algo.train_random_action_prob = 0.0
        #cfg.algo.valid_random_action_prob = 0.0


        cfg.algo.num_from_policy = 64
        cfg.algo.num_from_dataset= 0

    def setup_task(self):
        # The task we created above
        #self.task = MoleculeTask(dataset=self.training_data)
        self.task = MoleculeTask(data_tuple=self.data_tuple)

    #def setup_data(self):
        #self.training_data=VanillaDataset(train=True,data_frame=self.dataframe)
        #self.test_data=VanillaDataset(train=False,data_frame=self.dataframe)

    #def setup(self):
        #super().setup()
        #self.training_data.setup(self.task,self.ctx)
        #self.test_data.setup(self.task,self.ctx)

    def setup_env_context(self):
        # The per-atom generation context
        if self.fragments == None:
            print("Building molecules using atoms ...")
            self.ctx = MolBuildingEnvContext(
                ["C","N","O", "F", "S"], #["C","N","O", "F", "P", "S"],
                max_nodes=self.cfg.algo.max_nodes,  # Limit the number of atoms
                num_cond_dim=1,  # As per sample_conditional_information, this will be torch.ones((n, 1))
                charges=[0],  # disable charge
                chiral_types=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED],  # disable chirality
                num_rw_feat=0, #how many features are associated with each node during the random walk process. 
                expl_H_range=[0],# [0,1]
            )
        else:
            print("Building molecules using fragments ...")
            self.ctx = FragMolBuildingEnvContext(
                max_frags=self.cfg.algo.max_nodes,
                fragments=self.fragments
            )



            



