import json
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath('//home/mdavood/SPECTER/SPECTER2/scirepeval/evaluation'))  # Path to evaluation dir

def load_embeddings_from_jsonl(path):
    emb = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj["doc_id"])
            emb[pid] = np.array(obj["embedding"], dtype=np.float32)
    return emb

from typing import Union, Dict

import logging
import os
import datasets
import numpy as np
from tqdm import tqdm

from embeddings_generator import EmbeddingsGenerator
from encoders import Model
from eval_datasets import SimpleDataset
from evaluator import IREvaluator
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ReviewerMatchingEvaluator(IREvaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple],
                 reviewer_metadata: Union[str, tuple], model: Model,
                 metrics: tuple = ("P_5", "P_10"), batch_size: int = 16, fields: list = None):
        super(ReviewerMatchingEvaluator, self).__init__(name, meta_dataset, test_dataset, model, metrics, SimpleDataset,
                                                        batch_size, fields, )
        self.reviewer_metadata = reviewer_metadata

    def evaluate(self, embeddings, **kwargs):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if type(self.test_dataset) == str and os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("json",
                data_files={
                    "test_hard": f"{self.test_dataset}/test_hard_qrel.jsonl",
                    "test_soft": f"{self.test_dataset}/test_soft_qrel.jsonl"
                })
        else:
            split_dataset = datasets.load_dataset(
                self.test_dataset[0], self.test_dataset[1],
                trust_remote_code=False
            )

        logger.info(f"Loaded {len(split_dataset['test_hard'])} test query-candidate pairs for hard and soft tests")

        # If embeddings is a path, load your JSONL
        if isinstance(embeddings, str):
            embeddings = load_embeddings_from_jsonl(embeddings)

        qrels_hard = self.get_qc_pairs(split_dataset["test_hard"])
        qrels_soft = self.get_qc_pairs(split_dataset["test_soft"])

        preds = self.retrieval(embeddings, qrels_hard)

        results = {f"hard_{k}": v for k, v in self.calc_metrics(qrels_hard, preds).items()}
        results.update({f"soft_{k}": v for k, v in self.calc_metrics(qrels_soft, preds).items()})

        self.print_results(results)
        return results

    def retrieval(self, embeddings, qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        logger.info("Loading reviewer metadata...")
        if type(self.reviewer_metadata) == str and os.path.isdir(self.reviewer_metadata):
            reviewer_dataset = datasets.load_dataset("json", data_files={
                "metadata": f"{self.reviewer_metadata}/reviewer_metadata.jsonl"})["metadata"]
        else:
            reviewer_dataset = datasets.load_dataset(self.reviewer_metadata[0], self.reviewer_metadata[1],
                                                     split="metadata")
        logger.info(f"Loaded {len(reviewer_dataset)} reviewer metadata")
        reviewer_papers = {d["r_id"]: d["papers"] for d in reviewer_dataset}

        run = dict()
        for qid in tqdm(qrels):
            query = np.array([embeddings[qid]])
            cand_papers = {cid: np.array([embeddings[pid] for pid in reviewer_papers[cid]]) for cid in qrels[qid] if
                           cid in reviewer_papers}
            scores = {cid: cosine_similarity(cand_papers[cid], query).flatten() for cid in cand_papers}
            sorted_scores = {cid: sorted(scores[cid], reverse=True) for cid in scores}
            run[qid] = {cid: float(np.mean(sorted_scores[cid][:3])) for cid in sorted_scores}
        return run
    

meta_dataset      = ("allenai/scirepeval",      "paper_reviewer_matching")
test_dataset      = ("allenai/scirepeval_test", "paper_reviewer_matching")
reviewer_metadata = ("allenai/scirepeval_test", "reviewers")

# Dummy model; ReviewerMatchingEvaluator uses precomputed embeddings,
# so the Model instance usually isn't used inside evaluate() here.
dummy_model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=True)

evaluator = ReviewerMatchingEvaluator(
    name="Paper-Reviewer Matching",
    meta_dataset=meta_dataset,
    test_dataset=test_dataset,
    reviewer_metadata=reviewer_metadata,
    model=dummy_model,
    metrics=("P_5", "P_10"),
    batch_size=16,
    fields=None,
)

# Path to your embeddings JSONL
emb_path = "/home/mdavood/SPECTER/SPECTER2/scirepeval/embeddings-prm.json"  # with {"doc_id": ..., "embedding": [...]}

results = evaluator.evaluate(embeddings=emb_path)
print(results)
