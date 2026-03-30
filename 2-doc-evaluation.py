import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from evaluation.encoders import Model
from evaluation.embeddings_generator import EmbeddingsGenerator
from evaluation.eval_datasets import SimpleDataset  # or your own dataset wrapper

# 1. Make dataset object for 2 texts (title+abstract)
docs = [
    {"doc_id": "d1", "title": "SciRepEval: A Multi-Format Benchmark for Scientific Document Representations", "abstract": "Learned representations of scientific documents can serve as valuable input features for downstream tasks without further fine-tuning. However, existing benchmarks for evaluating these representations fail to capture the diversity of relevant tasks. In response, we introduce SciRepEval, the first comprehensive benchmark for training and evaluating scientific document representations. It includes 24 challenging and realistic tasks, 8 of which are new, across four formats: classification, regression, ranking and search. We then use this benchmark to study and improve the generalization ability of scientific document representation models. We show how state-of-the-art models like SPECTER and SciNCL struggle to generalize across the task formats, and that simple multi-task training fails to improve them. However, a new approach that learns multiple embeddings per document, each tailored to a different format, can improve performance. We experiment with task-format-specific control codes and adapters and find they outperform the existing single-embedding state-of-the-art by over 2 points absolute. We release the resulting family of multi-format models, called SPECTER2, for the community to use and build on."},
    {"doc_id": "d2", "title": "SPECTER: Document-level Representation Learning using Citation-informed Transformers", "abstract": "Representation learning is a critical ingredient for natural language processing systems. Recent Transformer language models like BERT learn powerful textual representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power. For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks. We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning. Additionally, to encourage further research on document-level models, we introduce SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. We show that SPECTER outperforms a variety of competitive baselines on the benchmark."},
]


model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=True)
embeddings = model(docs)
embeddings = embeddings.detach().cpu().numpy()
# gen = EmbeddingsGenerator(docs, model)
# emb = gen.generate_embeddings(save_path=None)
print(embeddings[0])

sim = float(cosine_similarity(
    embeddings[0].reshape(1, -1),
    embeddings[1].reshape(1, -1)
)[0,0])
print("cosine similarity", sim)