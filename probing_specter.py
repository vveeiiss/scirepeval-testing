import numpy as np
import torch
from evaluation.encoders import Model
import matplotlib
matplotlib.use("Agg")  # backend for environments without display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def normalize_texts(docs):
    """Convert input format (dicts with title+abstract) into plain text strings."""
    normalized = []
    for d in docs:
        if isinstance(d, dict):
            title = d.get("title", "")
            abstract = d.get("abstract", "")
            normalized.append(f"{title} \n\n{abstract}".strip())
        elif isinstance(d, str):
            normalized.append(d)
        else:
            raise ValueError("Each input item must be a dict or a string")
    return normalized


def tokenize_texts(model, texts, max_length=512):
    return model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)


def token_attention_scores(model, text, layer=-1):
    # normalize possible dict input to text
    if isinstance(text, dict):
        text = normalize_texts([text])[0]
    tokenized = tokenize_texts(model, [text])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    outputs = model.encoder(
        **tokenized,
        output_attentions=True,
        return_dict=True,
    )

    attentions = outputs.attentions[layer].detach().cpu().numpy()  # [batch, n_heads, seq_len, seq_len]
    # average across heads
    avg_attn = attentions[0].mean(axis=0)  # [seq_len, seq_len]
    cls_to_tokens = avg_attn[0]  # CLS token -> all tokens (including CLS)
    token_ids = tokenized["input_ids"][0].detach().cpu().tolist()
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)

    return tokens, avg_attn, cls_to_tokens


def plot_token_attention(tokens, cls_attention, top_k=20, output_path="token_attention.png"):
    # Limit to top_k tokens for readability
    top_idx = np.argsort(cls_attention)[::-1][:top_k]
    top_tokens = [tokens[int(i)] for i in top_idx]
    top_scores = cls_attention[top_idx]

    plt.figure(figsize=(min(12, len(top_tokens) * 0.4), 4))
    plt.barh(range(len(top_tokens)), top_scores, color="steelblue")
    plt.yticks(range(len(top_tokens)), top_tokens)
    plt.xlabel("CLS Attention Weight")
    plt.ylabel("Token")
    plt.title("Top token importance from CLS attention (layer)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved token attention plot to {output_path}")
    plt.close()


def plot_attention_heatmap(tokens, attention_matrix, output_path="attention_heatmap.png", top_k=40):
    # attention_matrix: [seq_len, seq_len]
    # choose top_k by mean attention so the plot is readable
    top_idx = np.argsort(np.mean(attention_matrix, axis=0))[::-1][:top_k]
    sub_tokens = [tokens[i] for i in top_idx]
    sub_mat = attention_matrix[np.ix_(top_idx, top_idx)]

    plt.figure(figsize=(min(14, len(sub_tokens) * 0.3), min(14, len(sub_tokens) * 0.3)))
    sns.heatmap(sub_mat, xticklabels=sub_tokens, yticklabels=sub_tokens, cmap="mako", cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("CLS attention heatmap (top tokens)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved attention heatmap to {output_path}")
    plt.close()


def attn_rollout(layer_attentions, start_layer=0):
    """Compute attention rollout from list of per-layer attention matrices."""
    # layer_attentions: list of [n_heads, seq_len, seq_len], where each is a layer
    if len(layer_attentions) == 0:
        raise ValueError("No attention layers provided for rollout")

    seq_len = layer_attentions[0].shape[-1]
    rollout = np.eye(seq_len, dtype=np.float32)

    for layer_idx, layer_attn in enumerate(layer_attentions[start_layer:], start=start_layer):
        # average heads if present
        if layer_attn.ndim == 3:
            layer_mat = layer_attn.mean(axis=0)
        elif layer_attn.ndim == 2:
            layer_mat = layer_attn
        else:
            raise ValueError("Unexpected attention shape in layer %d: %s" % (layer_idx, layer_attn.shape))

        # add residual and normalize rows
        layer_mat = layer_mat + np.eye(seq_len, dtype=np.float32)
        layer_mat = layer_mat / layer_mat.sum(axis=-1, keepdims=True)
        rollout = rollout.dot(layer_mat)

    return rollout


def plot_attention_matrix(tokens, attention_matrix, title, output_path="attention_matrix.png", top_k=40):
    n = len(tokens)
    if n == 0:
        raise ValueError("No tokens provided for attention matrix plot")

    top_k = min(top_k, n)
    top_idx = np.argsort(np.mean(attention_matrix, axis=0))[::-1][:top_k]
    top_tokens = [tokens[i] for i in top_idx]
    top_mat = attention_matrix[np.ix_(top_idx, top_idx)]

    plt.figure(figsize=(min(14, top_k * 0.3), min(14, top_k * 0.3)))
    sns.heatmap(top_mat, xticklabels=top_tokens, yticklabels=top_tokens, cmap="mako", cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved attention matrix plot to {output_path}")
    plt.close()


def probe_model(model, texts):
    texts = normalize_texts(texts)
    tokenized = tokenize_texts(model, texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    # Use the raw encoder for hidden states/attentions
    outputs = model.encoder(
        **tokenized,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )

    hidden_states = [h.detach().cpu().numpy() for h in outputs.hidden_states]
    attentions = [a.detach().cpu().numpy() for a in outputs.attentions]

    return {
        "cls_emb": outputs.last_hidden_state[:, 0, :].detach().cpu().numpy(),
        "hidden_states": hidden_states,
        "attentions": attentions,
        "tokenized": tokenized,
    }

    # CLS embedding (manual from encoder), same as pooler in Model.__call__
    cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    hidden_states = [h.detach().cpu().numpy() for h in outputs.hidden_states]
    attentions = [a.detach().cpu().numpy() for a in outputs.attentions]

    return {
        "cls_emb": cls_emb,
        "hidden_states": hidden_states,
        "attentions": attentions,
    }


def attention_entropy(attention_vec):
    # attention_vec is a 1d probability-like array over tokens
    att = np.asarray(attention_vec, dtype=np.float64)
    att = att / att.sum()
    # clip to avoid log(0)
    att = np.clip(att, 1e-12, 1.0)
    return -np.sum(att * np.log(att))


def analyze_layers_8_to_12(model, texts, results, out_prefix="layer8_12"):
    # results must contain hidden_states and attentions from probe_model
    layers = list(range(8, min(13, len(results["attentions"]))))
    insights = []

    for i, layer in enumerate(layers):
        attn = results["attentions"][layer]  # [batch, n_heads, seq_len, seq_len]
        attn_single = attn[0]
        avg_head = attn_single.mean(axis=0)  # [seq_len, seq_len]
        cls_attn = avg_head[0]
        cls_tokens = model.tokenizer.convert_ids_to_tokens(results["tokenized"]["input_ids"][0].cpu().tolist())

        # measure entropy of CLS attention in this layer
        ent = attention_entropy(cls_attn)

        # top 10 tokens for CLS attention this layer
        top_idx = np.argsort(cls_attn)[::-1][:10]
        top_tokens = [cls_tokens[idx] for idx in top_idx]
        top_scores = cls_attn[top_idx]

        insights.append({
            "layer": layer,
            "entropy": float(ent),
            "top_tokens": top_tokens,
            "top_scores": top_scores.tolist(),
        })

    # no longer saving per-layer attention plots (kept for brevity and speed)


    # CLS embedding similarity across layers 8-12 (numeric only, no plot)
    cls_vecs = [results["hidden_states"][l][0, 0, :] for l in layers]
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(np.stack(cls_vecs, axis=0))

    return insights, sim


def mean_cosine_by_layer(hidden_states, query_idx=0, candidate_idx=1):
    from sklearn.metrics.pairwise import cosine_similarity

    assert len(hidden_states[0]) > max(query_idx, candidate_idx)
    scores = []
    for layer in hidden_states:
        q = layer[query_idx : query_idx + 1, 0, :]
        c = layer[candidate_idx : candidate_idx + 1, 0, :]
        scores.append(float(cosine_similarity(q, c)[0, 0]))
    return scores


def run_document_probe(model, doc, idx, device="cpu", out_prefix="doc"):
    """Run token+layer probing for one document and return insights."""
    doc_text = normalize_texts([doc])[0]
    tokens, attention_matrix, cls_attention = token_attention_scores(model, doc_text, layer=-1)

    plot_token_attention(tokens, cls_attention, top_k=20, output_path=f"{out_prefix}{idx}_token_attention.png")
    plot_attention_heatmap(tokens, attention_matrix, output_path=f"{out_prefix}{idx}_attention_heatmap.png", top_k=40)

    single_input = model.tokenizer([doc_text], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outs = model.encoder(**single_input, output_hidden_states=True, output_attentions=True, return_dict=True)

    per_layer_results = {
        "hidden_states": [h.detach().cpu().numpy() for h in outs.hidden_states],
        "attentions": [a.detach().cpu().numpy() for a in outs.attentions],
        "tokenized": single_input,
    }

    layer_insights, layer_similarity = analyze_layers_8_to_12(model, [doc_text], per_layer_results, out_prefix=f"{out_prefix}{idx}_layer8_12")

    # optionally keep rollout for future use if needed locally
    per_layer_attn = [a.detach().cpu().numpy()[0] for a in outs.attentions]
    rollout = attn_rollout(per_layer_attn, start_layer=0)

    return {
        "doc_text": doc_text,
        "tokens": tokens,
        "attention_matrix": attention_matrix,
        "cls_attention": cls_attention,
        "layer_insights": layer_insights,
        "layer_similarity": layer_similarity,
        "full_rollout": rollout,
    }


def main():
    texts = [
        {"title": "SciRepEval: A Multi-Format Benchmark for Scientific Document Representations", "abstract": "Learned representations of scientific documents can serve as valuable input features for downstream tasks without further fine-tuning. However, existing benchmarks for evaluating these representations fail to capture the diversity of relevant tasks. In response, we introduce SciRepEval, the first comprehensive benchmark for training and evaluating scientific document representations. It includes 24 challenging and realistic tasks, 8 of which are new, across four formats: classification, regression, ranking and search. We then use this benchmark to study and improve the generalization ability of scientific document representation models. We show how state-of-the-art models like SPECTER and SciNCL struggle to generalize across the task formats, and that simple multi-task training fails to improve them. However, a new approach that learns multiple embeddings per document, each tailored to a different format, can improve performance. We experiment with task-format-specific control codes and adapters and find they outperform the existing single-embedding state-of-the-art by over 2 points absolute. We release the resulting family of multi-format models, called SPECTER2, for the community to use and build on."},
        {"title": "SPECTER: Document-level Representation Learning using Citation-informed Transformers", "abstract": "Representation learning is a critical ingredient for natural language processing systems. Recent Transformer language models like BERT learn powerful textual representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power. For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks. We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning. Additionally, to encourage further research on document-level models, we introduce SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. We show that SPECTER outperforms a variety of competitive baselines on the benchmark."},
        {"title": "Braess's paradox in tandem-running ants: When shortest path is not the quickest", "abstract": "Braess's paradox -- where adding network capacity increases travel time -- is typically attributed to selfish agents. Although eusocial colonies maximize collective fitness, we find experimentally that \emph{Diacamma indicum} ants exhibit this paradox: Leaders favour the shortest path even when it slows the colony. We present a quantitative model of the exploration-exploitation trade-off, demonstrating that evolutionary forces selecting for shortest-path identification can force suboptimal global states. This proves the paradox can emerge in highly cooperative systems without individual selfishness."},
    ]

    model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []

    for idx, doc in enumerate(texts, start=1):
        print(f"\n=== Document {idx} probe analysis ===")
        result = run_document_probe(model, doc, idx, device=device, out_prefix="doc")

        for x in result["layer_insights"]:
            print(f"Layer {x['layer']}: entropy={x['entropy']:.4f}, top={x['top_tokens']}")

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    main()
