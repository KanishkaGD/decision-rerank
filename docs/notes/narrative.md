# The narrative of the arXiv paper

Written 2026-10-14 at the owner's request ("a clear narrative; too many buts"). Every text part of the paper
(abstract, introduction, Sects. 5-6, conclusion), the teaser figure and the project report follow this page. Numbers
are those of the generated tables (`paper/tables/`); quote no others.

## Title

**System One Models Are Not Better Rerankers** (no subtitle). It states the finding (RQ1: never better on the same
backbone; worse up to 9B, a tie at 27B), it is short, and it reads as a claim, not a question.

## Scope

Our 27B reranker fine-tune is in the paper (owner, 2026-10-19, decision 24, after a same-day removal was reverted):
it gives RQ1 a measured answer at 27B, which the title needs.

## One sentence

On the same backbone, decision fine-tuning never makes a better reranker than plain reranker fine-tuning, and the
calibration advantage it is advertised for depends on how the question is worded.

## Three punchlines (in this order, in this wording, everywhere)

1. **Reranker fine-tuning beats decision fine-tuning.** On the same backbone, the reranker fine-tune ranks better on
   every backbone from 0.6B to 9B (by 5 to 29 points nDCG@10; significant on 48 of the 60 pairs of reranker fine-tune and collection (6 reranker fine-tunes up to 9B, Ours-9B
   included), never
   significantly worse; also on the zero-shot collections) and ties at 27B (our fine-tune 59.6 vs. Kev-27B 60.1).
   Decision fine-tuning is never better.
2. **Up to 9B, decision fine-tuning adds nothing over the untuned model.** Within -2.5 and +0.7 points of simply
   asking the untuned model the same question; reranker fine-tuning adds 6 to 30. At 27B, both kinds of fine-tuning
   add about the same (Kev-27B +3.7, our fine-tune +3.2).
3. **The calibration advantage depends on the question.** With a question that matches the relevance labels ("Does
   the passage answer the query?"), decision models from 4B on are better calibrated than the published reranker
   fine-tunes; with S1Rank's looser question they lose that advantage, while their ranking hardly moves. The untuned
   models gain as much from the wording. Calibration does not make up for a worse ranking: up to 9B, a reranker
   fine-tune keeps the better set at a fixed threshold. (Reworded 2026-10-19, decision 25: "comes from the question,
   not from the model" went too far, since decision fine-tuning does improve on the untuned model with the same
   question, e.g. Qwen3-8B .231 -> Kev-8B .113; and "calibration does not buy better passage selection" ignored
   27B, where the rankings tie and the better calibrated Kev-27B keeps the better set.)

Closing line: **If you need a reranker, fine-tune a reranker.**

## What is not a punchline (reported in the sections, not in abstract/introduction/conclusion)

- At 27B, Kev-27B is better calibrated than our reranker fine-tune with our question (ECE .076 vs .110) and keeps the
  better set at p >= 0.5 (57.2 vs 48.3). Reported in Sect. 6 as a plain result, without hedging the headline: it is
  question-dependent like every calibration advantage (with S1Rank's question Kev-27B has ECE .156).
- Count test, post-hoc scaling (temperature/Platt on MS MARCO dev), regret, joint vs. isolated: results in their
  sections; at most one clause in the introduction.
- The numbers of the comparisons with the published reranker fine-tunes' calibration: in Sect. 6.
- Why 27B differs (Kev-27B: full fine-tune of the chat model, longer training states; Clef at 27B adds nothing):
  Sect. 7, discussion.

## Style

Short, assertive sentences; claims first, evidence second. No "however"/"but" chains in abstract, introduction or
conclusion: a qualification either belongs to the claim (then the claim is worded to include it, e.g. "up to 9B")
or goes to the results sections. Plain words, no dash asides. "Backbone" for the pretrained model of a size, "untuned
model" for the released chat model asked zero-shot (not "base model", which LLM readers take for the pretrained
checkpoint; decision 25), "decision model" after the first mention of "System One models". All terms are defined
once, in Sect. 3 ("Terms").
