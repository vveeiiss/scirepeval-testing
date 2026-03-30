try:
    # Optional dependency: only needed when using adapter-specific model variants.
    # This script currently runs `variant="default"`, so we don't strictly require `adapters`.
    from adapters import AutoAdapterModel  # noqa: F401
except Exception:
    AutoAdapterModel = None
from datasets import load_dataset

from transformers import AutoTokenizer
import os, sys
import torch
print("torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())


ds = load_dataset("allenai/scirepeval", "paper_reviewer_matching")
ds = load_dataset("allenai/scirepeval_test", "paper_reviewer_matching")


sys.path.insert(0, os.path.abspath('//home/mdavood/SPECTER/SPECTER2/scirepeval/evaluation'))  # Path to evaluation dir

from eval_datasets import SimpleDataset
from eval_datasets import IRDataset  
from evaluator import Evaluator
from encoders import Model

adapters_dict = {"[CLF]": "allenai/scirepeval_adapters_clf", "[QRY]": "allenai/scirepeval_adapters_qry", "[RGN]": "allenai/scirepeval_adapters_rgn", "[PRX]": "allenai/scirepeval_prx"}
#model = Model(variant="adapters", base_checkpoint="malteos/scincl", adapters_load_from=adapters_dict, all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=True)

model.task_id = "[PRX]" #OR "[RGN]"/"[PRX]"/{{"query": "[QRY]", "candidates": "[PRX]"}}}

print(torch.cuda.is_available())

dataset = ("allenai/scirepeval", "paper_reviewer_matching") #OR path like "scirepeval/biomimicry/test.json"
evaluator = Evaluator(name="paper_reviewer_matching", meta_dataset= dataset,  dataset_class=SimpleDataset, model=model, batch_size=32, fields=["title", "abstract"], key="doc_id")
#embeddings = evaluator.generate_embeddings(save_path="embeddings-prm.json")
print("prm set done")
dataset = ("allenai/scirepeval", "same_author") #OR path like "scirepeval/biomimicry/test.json"
evaluator = Evaluator(name="same_author", meta_dataset= dataset,  dataset_class=IRDataset, model=model, batch_size=32, fields=["title", "abstract"], key="doc_id")
#embeddings = evaluator.generate_embeddings(save_path="embeddings-2doc.json")
print("first set done")

dataset_1 = ("allenai/scirepeval", "high_influence_cite") #OR path like "scirepeval/biomimicry/test.json"
evaluator_1 = Evaluator(name="high_influence_cite", meta_dataset= dataset,  dataset_class=IRDataset, model=model, batch_size=32, fields=["title", "abstract"], key="doc_id")
embeddings_1 = evaluator.generate_embeddings(save_path="embeddings-2doc-citetion.json")
print("second set done")

