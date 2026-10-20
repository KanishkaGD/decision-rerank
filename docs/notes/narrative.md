# The narrative of the arXiv paper

Written 2026-10-14 at the owner's request ("a clear narrative; too many buts"). Every text part of the paper
(abstract, introduction, Sects. 5-7, conclusion), the teaser figure and the project report follow this page. Numbers
are those of the generated tables (`paper/tables/`); quote no others.

## Title

**System One Models Are Not Better Rerankers** (no subtitle). It states the finding (RQ1: never better on the same
backbone; worse up to 9B, a tie at 27B), it is short, and it reads as a claim, not a question.

## Scope

Our 27B reranker fine-tune is in the paper (owner, 2026-10-19, decision 24, after a same-day removal was reverted):
it gives RQ1 a measured answer at 27B, which the title needs.

## One sentence

On the same backbone, decision fine-tuning never makes a better reranker than plain reranker fine-tuning, and the
calibration it is advertised for comes from how the question is worded, not from the model.

## Three punchlines (in this order, in this wording, everywhere)

1. **Reranker fine-tuning beats decision fine-tuning.** On the same backbone, the reranker fine-tune ranks better on
   every backbone from 0.6B to 9B (by 5 to 29 points nDCG@10; significant on 48 of 60 backbone-collection pairs, never
   significantly worse; also on the zero-shot collections) and ties at 27B (our fine-tune 59.6 vs. Kev-27B 60.1).
   Decision fine-tuning is never better.
2. **Up to 9B, decision fine-tuning adds nothing over the base model.** Within -2.5 and +0.7 points of simply asking
   the untuned base model the same question; reranker fine-tuning adds 6 to 30. At 27B, both kinds of fine-tuning
   add about the same (Kev-27B +3.7, our fine-tune +3.2).
3. **The calibration advantage comes from the question, not from the model.** With a question that matches the
   relevance labels ("Does the passage answer the query?"), decision models are well calibrated; with S1Rank's
   looser question the same models lose that advantage over reranker fine-tunes, while their ranking hardly moves.
   The untuned base models gain as much from the wording. And calibration does not buy better passage selection: up
   to 9B, a reranker fine-tune keeps the better set at a fixed threshold.

Closing line: **If you need a reranker, fine-tune a reranker.**

## What is not a punchline (reported in the sections, not in abstract/introduction/conclusion)

- At 27B, Kev-27B is better calibrated than our reranker fine-tune with our question (ECE .076 vs .110) and keeps the
  better set at p >= 0.5 (57.2 vs 48.3). Reported in Sect. 6 as a plain result, without hedging the headline: it is
  question-dependent like every calibration advantage (with S1Rank's question Kev-27B has ECE .156).
- Count test, post-hoc scaling (temperature/Platt on MS MARCO dev), regret, joint vs. isolated: results in their
  sections; at most one clause in the introduction.
- Comparisons with the *published* reranker fine-tunes' calibration: in Sect. 6, not in the punchlines.

## Style

Short, assertive sentences; claims first, evidence second. No "however"/"but" chains in abstract, introduction or
conclusion: a qualification either belongs to the claim (then the claim is worded to include it, e.g. "up to 9B")
or goes to the results sections. Plain words, no dash asides. "Backbone" for the pretrained model of a size, "base
model" for the untuned variant (as in the ECIR version).
