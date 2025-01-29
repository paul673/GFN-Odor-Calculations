import sys
import os
print(os.getcwd())
path_to_top_dir = "../"
# Dynamically add the `project` directory to the path
sys.path.append(os.path.abspath(path_to_top_dir))
print(os.listdir())

from tqdm import tqdm
# Utils
import warnings

# Plotting
from rdkit.Chem.Draw import MolsToGridImage
import matplotlib.pyplot as plt
import seaborn as sns
from pycirclize import Circos

# File management
import json
import numpy as np
import pandas as pd


# GFlowNET
import gflownet
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.config import init_empty, Config
from scent_gfn.molecule import MoleculeTask

# OpenPOM
from pom_models.functions import fragance_propabilities_from_smiles

# Others
from tensorboard.backend.event_processing import event_accumulator
import torch
from rdkit import Chem

from matplotlib.lines import Line2D
import gc

def calc_porperties(arr):
    return dict(
        arr=arr,
        average = float(np.mean(arr)),
        median = float(np.median(arr)),
        quantile25 = float(np.quantile(arr,0.25)),
        quantile75 = float(np.quantile(arr,0.75)),
        variance = float(np.var(arr, ddof=1)),  # ddof=1 for sample variance
        minimum = float(np.min(arr)),
        maximum = float(np.max(arr))
    )

def save_size(savename,number):
    
    # Model class must be defined somewhere
    state = torch.load(os.path.join(savename, "model_final_save.pt"), weights_only=False)
    #model = statemodel.eval()
    #state["models_state_dict"]

    env_ctx = state["env_ctx"]


    model = gflownet.models.graph_transformer.GraphTransformerGFN(state["env_ctx"],state["cfg"])
    model.load_state_dict(state["models_state_dict"][0])
    model.eval()
    algo = TrajectoryBalance(GraphBuildingEnv(),state["env_ctx"],state["cfg"])
    #task = MoleculeTask(data_tuple=(params["reward_func"], params["similarity"],params["large_mol_pen"], target, params["beta"]))
    trajs = algo.create_training_data_from_own_samples(model, number)

    objs = [env_ctx.graph_to_obj(i['result']) for i in trajs]
    del model
    del algo
    del trajs
    gc.collect()
    atom_length_lst = []
    smiles_list=[]
    for obj in objs:
        num_of_atoms = obj.GetNumAtoms()
        atom_length_lst.append(num_of_atoms)
        smiles_list.append(Chem.MolToSmiles(obj))

    result_dict= calc_porperties(atom_length_lst)
    result_dict["smiles"]=smiles_list
    return result_dict
    



directory = "../results"
vanillasaves=[]
pinesaves=[]
# Iterate over files in directory
for name in os.listdir(directory):
    # Open file
    #print(name)
    if "vanilla" in name:
        vanillasaves.append(name)
    elif "pine" in name:
        pinesaves.append(name)



vanillalst = ["", "vanilla8","vanilla9", "vanilla10","", "vanilla1", "vanilla2", "vanilla11", "vanilla12", "vanilla3","vanilla4"]  

vanillanames={
    "vanilla1": "OpenPOM, Cosine",
    "vanilla2": "Structure, Cosine",
    "vanilla3": "OpenPOM, Cosine, Exp.pen",
    "vanilla4": "OpenPOM, Cosine, Hard.pen",
    "vanilla5": "",
    "vanilla6": "",
    "vanilla7": "",
    "vanilla8": "OpenPOM, Cosine",
    "vanilla9": "Structure, Cosine",
    "vanilla10": "Structure, Cosine, Beta",
    "vanilla11": "OpenPOM, Tanimoto",
    "vanilla12": "Structure, Tanimoto",
}
pinenames = {
    "pine1": "OpenPOM, Cosine",
    "pine2": "OpenPOM, Cosine, Hard.pen"
}
pinelst= ["", "pine1", "pine2"]  

def find_save(van_str):
    for save in vanillasaves:
        if van_str==save.split("_")[0]:
            return save
    return "not found"
def find_save_pine(van_str):
    for save in pinesaves:
        if van_str==save.split("_")[0]:
            return save
    return "not found"

directory = "../results"
#results = {}
# Iterate over files in directory
print("Initialize")
"""alist =['vanilla8_2024-12-05__22_21_26',
 'vanilla9_2024-12-06__20_41_11',
 'vanilla10_2024-12-07__10_47_14',
 'vanilla1_2024-12-03__18_44_11',
 'vanilla2_2024-12-03__20_50_10',
 'vanilla11_2024-12-13__20_56_37',
 'vanilla12_2024-12-13__22_27_31',
 'vanilla3_2024-12-03__22_12_54',
 'vanilla4_2024-12-04__09_23_46',
 'pine1_2024-12-03__14_19_29',
 'pine2_2024-12-08__20_52_30']"""
alist =sys.argv[1:]
print(alist)

for name in tqdm(alist):
    savename = f"c:/Users/paulj/Desktop/prosjekt/pom_cpu-model/results/{name}"
    resultd = save_size(savename,250)
    #resultd["name"]=name

    with open(f"../images/json/{name}.json","w+")as file:
        json.dump(resultd,file)
    del resultd
    gc.collect()

#with open("../images/size.json","w+")as file:
   # json.dump(results,file,)


