# Plan d'entrainement complet : nanoInstruct (ex nanochat-attnres)

**Important** : Ce projet n'est PAS un modele de chat. C'est un modele **instruct** specialise pour repondre uniquement a partir d'un contexte fourni (grounded QA, extraction structuree, classification, synthese).

**Objectif final** : un LLM bilingue FR/EN **grounded** avec plusieurs Talkers specialises pour differentes taches.

---

## Architecture cible

```
┌───────────────────────────────────────────────────────────────────┐
│  REASONER (978M, Phase 1 + Phase 2 JEPA, puis FIGE)               │
│  Role : comprendre et raisonner sur le contexte                   │
│         Produit des pensees latentes (vecteurs continus)          │
│         Architecture : hybride conv-attention (SSSL) + AttnRes    │
└───────────────────────────────────────────────────────────────────┘
                           │
                           ▼ latents + contexte brut
         ┌─────────────────┼─────────────────────────┐
         ▼                 ▼                         ▼
┌──────────────┐  ┌──────────────┐        ┌──────────────────┐
│ Talker_QA    │  │ Talker_JSON  │  ...   │ Talker_Synthesis │
│ ~200M        │  │ ~150M        │        │ ~250M            │
│ Grounded QA  │  │ Extraction   │        │ Summarization    │
│ counterfact. │  │ structuree   │        │                  │
└──────────────┘  └──────────────┘        └──────────────────┘
```

---

## Phase 0 : Setup avec special tokens complets (AVANT Phase 1)

**Critique** : tous les special tokens doivent etre dans la tokenizer DES Phase 1 pour que leurs embeddings soient co-entraines avec le reste du modele.

### Special tokens (37 au total, instruct-only -- pas de chat)

Defini dans `nanochat/tokenizer.py` dans `SPECIAL_TOKENS` :

| Categorie | Tokens | Usage |
|-----------|--------|-------|
| Base | `<\|bos\|>` | Beginning of sequence |
| Context | `<\|context_start\|>`, `<\|context_end\|>` | Wrapper du contexte (RAG, grounded QA) |
| Input | `<\|input_start\|>`, `<\|input_end\|>` | Wrapper de l'instruction/question |
| Task | `<\|qa\|>`, `<\|extract_json\|>`, `<\|extract_triples\|>`, `<\|classify\|>`, `<\|summarize\|>` | Dispatchers Talker (5 taches) |
| Reasoning | `<\|think_start\|>`, `<\|think_end\|>`, `<\|answer_start\|>`, `<\|answer_end\|>`, `<\|no_answer\|>` | Chain-of-thought |
| Structured | `<\|json_start\|>`/`<\|json_end\|>`, `<\|triple_start\|>`/`<\|triple_end\|>`, `<\|class_start\|>`/`<\|class_end\|>`, `<\|summary_start\|>`/`<\|summary_end\|>` | Outputs specifiques par Talker |
| Code | `<\|code_start\|>`, `<\|code_end\|>` | Wrapper code dans contexte ou output |
| Meta | `<\|citation_start\|>`, `<\|citation_end\|>`, `<\|uncertain\|>` | Citations et incertitude |
| JEPA | `<\|latent\|>` | Placeholder pensee latente |
| Reserved | `<\|reserved_0\|>` ... `<\|reserved_7\|>` | 8 slots pour extension future |

### Format type d'un sample instruct

```
<|bos|>
<|context_start|>
[document(s) source ; peut contenir <|code_start|>...<|code_end|>]
<|context_end|>
<|qa|>                                  # task trigger
<|input_start|>[question]<|input_end|>
<|think_start|>
[raisonnement grounded au contexte uniquement]
<|think_end|>
<|answer_start|>
[reponse OU <|no_answer|> si pas dans le contexte]
<|answer_end|>
```

### Exemple counterfactual (test du grounding)

```
<|bos|>
<|context_start|>
La tour Eiffel a ete construite en 1889 a Tokyo par Gustave Eiffel.
<|context_end|>
<|qa|>
<|input_start|>Ou se trouve la tour Eiffel ?<|input_end|>
<|think_start|>
Selon le contexte, la tour Eiffel est a Tokyo.
Je dois suivre le contexte meme si ma memoire dit autrement.
<|think_end|>
<|answer_start|>
A Tokyo.
<|answer_end|>
```

### Setup Phase 0 (a faire AVANT Phase 1)

```bash
# 1. Sync nanochat/tokenizer.py avec les nouveaux SPECIAL_TOKENS sur le serveur
# 2. Retrain tokenizer (5-10 min)
uv run --no-sync python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

# 3. Restart Phase 1 from scratch
bash runs/train.sh
```

---

## Phase 1 : Pretraining base (~25 jours)

**Objectif** : acquerir le langage FR/EN + bases de raisonnement.

### Configuration

- **Architecture** : d32, ~978M params, hybride SSSL (24 conv + 8 attn), AttnRes, SwiGLU, GQA 4:1, FP8
- **Tokenizer** : 32768 vocab dont 37 special tokens (instruct-only)
- **Data** : curriculum 2-phases
  - Phase 1A (60% des steps) : general 65%, math 28%, reasoning 7%
  - Phase 1B (40% des steps) : general 40%, math 45%, reasoning 15%
- **Ratio** : `target-param-data-ratio=100` → ~93B tokens → ~181K steps
- **Save** : tous les 2000 steps, rotation a 1 checkpoint
- **Init** : tied init (angular alignment wte/lm_head) pour preparer JEPA

### Monitoring (wandb)

- `val/bpb` : val bits-per-byte
- `train/loss`, `train/tok_per_sec`, `train/mfu`
- `model/wte_lmhead_cos_sim` : tracker alignement (critique pour JEPA)
- `model/smear_lambda`, `model/backout_lambda` : scalaires apprenants
- `attnres/entropy_norm_mean` : diversite attention AttnRes
- `curriculum/general`, `curriculum/math`, `curriculum/reasoning` : weights courants

### Criteres de succes

- `val_bpb < 0.70`
- `core_metric > 0.25`
- `model/wte_lmhead_cos_sim > 0.4` (pour faciliter JEPA)

---

## Phase 2 : SFT format-aware leger (3-5 jours)

**But** : apprendre au modele le FORMAT instruct (les special tokens existent deja dans le vocab depuis Phase 1, il faut leur donner du contexte d'usage).

### Dataset

~50-100K samples synthetiques couvrant les 5 taches, formattes avec les special tokens.

### Fichiers

- `scripts/generate_format_sft.py` : generation dataset multi-task
- `scripts/sft_format.py` : training script (utilise `tokenizer.render_instruct_sample`)

---

## Phase 3 : JEPA SST -- Reasoner en latent (5-10 jours)

**But** : transformer le modele en Reasoner latent pur.

### Mecanisme

- **Loss** : scaled cosine distance (k=4) entre h_pred (student) et h_target (teacher EMA)
- **Teacher** : copie EMA (momentum 0.98)
- **L2 normalization** sur hidden states → hypersphere
- **Data** : meme mix Phase 1 (text continu, split en segments)

### Modifications code

- `nanochat/gpt.py` : ajouter `return_hidden=True` au forward
- `nanochat/jepa.py` : nouveau module (EMA, cosine loss, L2 norm)
- `scripts/jepa_sst.py` : script training Phase 3

### Checkpoint final

`reasoner_d32` → **fige** pour tout ce qui suit.

### Criteres de succes

- Cosine loss descend et se stabilise
- Rank des vecteurs latents > 100 (diversite preservee)
- Norm hidden ≈ 1 (L2 normalization OK)

---

## Phase 4 : Talkers specialises (parallele, 15-30 jours)

Le Reasoner est FIGE. Chaque Talker est entraine independamment.

### Architecture Talker generique

```python
# nanochat/talker.py (nouveau module)
class Talker(nn.Module):
    def __init__(self, n_layer=12, d_model=768, vocab_size=32768, latent_dim=1536):
        # Transformer decoder classique
        # Cross-attention sur latents du Reasoner
        # Cross-attention optionnelle sur contexte brut (pour grounding strict)
```

### 4a. Talker_QA (priorite 1)

- **Taille** : ~200M (12 layers, d_model=768)
- **Duree** : 5-7 jours
- **Dataset** : counterfactual SFT (faithful 60% + counterfactual 30% + no_answer 10%)
- **Volume cible** : 200K samples
- **Token trigger** : `<|qa|>`
- **Output wrapper** : `<|answer_start|>` ... `<|answer_end|>` ou `<|no_answer|>`

**Pipeline generation** (`scripts/generate_counterfactual_qa.py`) :
1. Extraction triplets (entite, relation, valeur) depuis Wikipedia + FineWeb2-HQ
2. Perturbation NLP (swap d'entites du meme type)
3. Formation samples faithful + counterfactual + no_answer

### 4b. Talker_JSON (priorite 2)

- **Token trigger** : `<|extract_json|>`
- **Output wrapper** : `<|json_start|>` ... `<|json_end|>`
- **Datasets** : REBEL, FewRel, synthetiques depuis Wikipedia
- **Taille** : ~150M, duree 3-5 jours

### 4c. Talker_Triples (priorite 3)

- **Token trigger** : `<|extract_triples|>`
- **Output wrapper** : `<|triple_start|>` ... `<|triple_end|>` (un wrapper par triplet)
- **Datasets** : REBEL, DocRED, TACRED
- **Taille** : ~150M, duree 3-5 jours

### 4d. Talker_Classify (priorite 4)

- **Token trigger** : `<|classify|>`
- **Output wrapper** : `<|class_start|>` ... `<|class_end|>`
- **Datasets** : AG News, MNLI, Allocine
- **Taille** : ~100M, duree 2-3 jours

### 4e. Talker_Summarize (priorite 5)

- **Token trigger** : `<|summarize|>`
- **Output wrapper** : `<|summary_start|>` ... `<|summary_end|>`
- **Datasets** : XSum, Orange Sum, arxiv-summarization
- **Taille** : ~250M, duree 4-6 jours

---

## Phase 5 : RL grounding (optionnel, 10-20 jours)

**But** : affiner Talker_QA avec rewards de grounding.

- **Reward** : verifieur LLM externe (check "reponse soutenue par le contexte ?")
- **Algo** : DPO d'abord (stable), puis GRPO/PPO si necessaire
- **Cible** : Talker_QA uniquement (Reasoner fige)

---

## Timeline globale

```
Semaine  0    : Phase 0 (setup tokens + retrain tokenizer)         1 jour
Semaine  1-4  : Phase 1 pretraining                                ~25 jours
Semaine  5    : Phase 2 SFT format-aware                           3-5 jours
Semaine  6-7  : Phase 3 JEPA SST                                   5-10 jours
Semaine  8-13 : Phase 4 Talkers (paralleles si plusieurs GPU)      15-30 jours
Semaine 14-17 : Phase 5 RL grounding (optionnel)                   10-20 jours

Total : ~12-17 semaines (3-4 mois)
```

---

## Inference : dispatcher multi-Talker

Script `scripts/chat_multitask.py` (nom historique, en realite c'est un dispatcher instruct) :

```python
def generate(task, context, query, n_thoughts=5):
    # Charge Reasoner une seule fois (global)
    talker = load_talker(task)  # qa, extract_json, extract_triples, classify, summarize

    # Reasoner produit des latents
    latents = reasoner.generate_latents(context, query, n_thoughts)

    # Talker decode (avec cross-attention sur latents + contexte brut)
    output = talker.generate(context, latents)
    return output
```

**Memoire** :
- Reasoner : ~2GB (fige)
- 1 Talker actif : ~300-500MB
- Total : ~2.5GB → tourne sur n'importe quel GPU moderne

---

## Datasets a preparer (sur DGX Spark en parallele de Phase 1)

| Talker | Datasets sources | Transformation | Volume cible |
|--------|------------------|----------------|--------------|
| QA | Wikipedia FR/EN, FineWeb2-HQ | Extraction triplets + perturbation NLP | 200K samples |
| JSON | REBEL, FewRel, auto Wikipedia | Formatage JSON avec schema | 100K samples |
| Triples | REBEL, DocRED, TACRED | Filtrage qualite + format | 150K samples |
| Classify | AG News, MNLI, Allocine | Unification format | 500K samples |
| Summarize | XSum, Orange Sum, arxiv | Filtrage longueurs | 100K samples |

---

## Fichiers a creer

### Scripts
- `scripts/generate_format_sft.py` -- Phase 2 datagen
- `scripts/sft_format.py` -- Phase 2 training
- `scripts/jepa_sst.py` -- Phase 3 training
- `scripts/generate_counterfactual_qa.py` -- Phase 4a datagen
- `scripts/generate_json_data.py` -- Phase 4b datagen
- `scripts/generate_triples_data.py` -- Phase 4c datagen
- `scripts/train_talker_qa.py` -- Phase 4a training
- `scripts/train_talker_json.py` -- Phase 4b training
- `scripts/train_talker_triples.py` -- Phase 4c training
- `scripts/train_talker_classify.py` -- Phase 4d training
- `scripts/train_talker_summarize.py` -- Phase 4e training
- `scripts/chat_multitask.py` -- inference dispatcher

### Modules
- `nanochat/talker.py` -- architecture Talker generique
- `nanochat/jepa.py` -- utilitaires EMA et cosine loss

### Modifications a faire
- `nanochat/gpt.py` -- ajouter `return_hidden=True` au forward (Phase 3)
- `nanochat/checkpoint_manager.py` -- support checkpoints Talker

### Modifications deja faites (Phase 0)
- `nanochat/tokenizer.py` -- 37 special tokens instruct-only, `render_instruct_sample`, `render_for_completion`
- `nanochat/engine.py` -- stop tokens base sur les outputs instruct, suppression chat-specific
- `nanochat/gpt.py` -- tied init wte/lm_head (preparation JEPA)
- `scripts/base_train.py` -- log `model/wte_lmhead_cos_sim`

### A nettoyer (chat-specific, dead code en mode instruct)
- `scripts/chat_sft.py`, `scripts/chat_rl.py`, `scripts/chat_eval.py`, `scripts/chat_cli.py`, `scripts/chat_web.py` -- utilisaient l'ancien format chat (user/assistant), ne fonctionneront plus tels quels

---

## Criteres de succes par phase

### Phase 1
- `val_bpb < 0.70`
- `core_metric > 0.25`
- `model/wte_lmhead_cos_sim > 0.4`

### Phase 3 (JEPA)
- Cosine loss SST descend et se stabilise
- Rank des latents > 100 (pas de collapse)
- Norm hidden ≈ 1

### Phase 4a (Talker_QA)
- Samples counterfactual : >95% de suivi du contexte vs memoire parametrique
- Samples unanswerable : >90% de `<|no_answer|>` correct
- Hallucination rate < 5%

### Phase 4b-e
- JSON : validite schema > 98%
- Triples : F1 > 0.7 sur REBEL test
- Classify : accuracy > 0.85 sur AG News
- Summarize : ROUGE-L > 0.35 sur XSum

---

## Pistes de recherche ouvertes (post-projet)

- Hard-tied embeddings + AttnRes : verifier la conjecture
- Recursive AttnRes : partage poids en profondeur, query-controlled loop count
- LCM concept prediction : predire embeddings de phrase
- Multimodal Talker : latents + image
- RL multi-task : optimiser tous les Talkers conjointement

---

## Note sur le nom

Le projet s'appelle officiellement **nanochat-attnres** (fork de karpathy/nanochat) mais sa nature reelle est **instruct only, jamais chat**. Un nom plus juste serait **nanoInstruct** ou **nanoGround** (pour grounded QA). Renommer le repo pourrait etre fait quand le code sera stable.
