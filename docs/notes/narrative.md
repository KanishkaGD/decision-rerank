# The narrative of the arXiv paper

Written 2026-10-14 at the owner's request ("a clear narrative; too many buts"). Every text part of the paper
(abstract, introduction, Sects. 5-7, conclusion), the teaser figure and the project report follow this page. Numbers
are those of the generated tables (`paper/tables/`); quote no others.

## Title

**System One Models Are Not Better Rerankers** (no subtitle). It states the finding of RQ1 on every backbone where
both kinds of fine-tuning exist (0.6B to 9B), it is short, and it reads as a claim, not a question.

## Scope (owner, 2026-10-19)

As in the ECIR version: no reranker fine-tune of the 27B backbone. Our 27B reranker fine-tune
(`qwen3.8-27b-reranker`) is not in the paper; its runs stay in `results/results.csv`. Ours-9B stays, as the control
for the published recipes. What stays from the arXiv revision: the question wording (our question for every result,
S1Rank's only in the ablation), the restored tables and the two-column ACM layout.

## One sentence

On the same backbone, decision fine-tuning does not make a better reranker than plain reranker fine-tuning, and the
calibration it is advertised for comes from how the question is worded, not from the model.

## Three punchlines (in this order, in this wording, everywhere)

1. **Reranker fine-tuning beats decision fine-tuning.** On every backbone where both exist (0.6B to 9B), the reranker
   fine-tune ranks better, by 5 to 29 points nDCG@10; significant on 48 of 60 backbone-collection pairs, never
   significantly worse; also on the zero-shot collections. At 27B, where no reranker fine-tune exists, Kev-27B is the
   best system in the study, but only 0.7 points ahead of reranker fine-tunes a third of its size.
2. **Up to 9B, decision fine-tuning adds nothing over the base model.** Within -2.5 and +0.7 points of simply asking
   the untuned base model the same question; reranker fine-tuning adds 6 to 30. Only Kev-27B clearly improves on its
   base model (+3.7).
3. **The calibration advantage comes from the question, not from the model.** With a question that matches the
   relevance labels ("Does the passage answer the query?"), decision models are well calibrated; with S1Rank's
   looser question the same models lose that advantage over reranker fine-tunes, while their ranking hardly moves.
   The untuned base models gain as much from the wording. And calibration does not buy better passage selection: a
   reranker fine-tune keeps the better set at a fixed threshold on every backbone where both exist.

Closing line: **If you need a reranker, fine-tune a reranker.**

## What is not a punchline (reported in the sections, not in abstract/introduction/conclusion)

- Count test, post-hoc scaling (temperature/Platt on MS MARCO dev), regret, joint vs. isolated: results in their
  sections; at most one clause in the introduction.
- Comparisons with the *published* reranker fine-tunes' calibration: in Sect. 6, not in the punchlines.
- The 27B backbone has no reranker fine-tune: say so plainly where 27B results are reported (Table 1 caption,
  Sect. 5, limitations), without speculation beyond the ECIR sentence (a gain as on the smaller backbones would put a
  27B reranker fine-tune ahead of Kev-27B).

## Style

Short, assertive sentences; claims first, evidence second. No "however"/"but" chains in abstract, introduction or
conclusion: a qualification either belongs to the claim (then the claim is worded to include it, e.g. "up to 9B")
or goes to the results sections. Plain words, no dash asides. "Backbone" for the pretrained model of a size, "base
model" for the untuned variant (as in the ECIR version).
