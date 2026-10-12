# Venue

**Current decision (owner, 2026-10-12): ECIR 2027 short paper, submission deadline 2026-10-12 23:59 AoE (2026-10-13 13:59 CEST)** (as stated
by the owner). Official call: https://www.ecir2027.co.uk/call-for-short-papers (read 2026-10-12): **up to 6 pages
plus references** (appendices count toward the 6 pages and go before the references), Springer LNCS proceedings
templates -- the paper's sources are already LNCS (`llncs.cls`, `splncs04.bst`) --, **double-anonymous review**
("fully anonymised"), EasyChair (conference id ecir27), notification 2026-12-07. A prior arXiv version is allowed, but
the submission must not cite it. Deadline: **12 October 2026, 23:59 AoE** (= 2026-10-13, 13:59 CEST); the abstract was registered before 5
October (owner, 2026-10-12). The
paper is written with what exists, plus at most minor experiments. Handoff: `docs/notes/paper-handoff.md`.

**arXiv version at the end of the week of 2026-10-12**, after the ECIR submission: the ECIR paper plus our 27B
reranker fine-tune next to the 9B one (`docs/intent/paper-focus.md` 15).

The WSDM decision below is superseded (kept as the record).

Decided by the owner on 2026-10-06 (open question 1 of the setup session).

**An arXiv preprint, then a WSDM 2027 short paper.** Call for papers: https://www.wsdm-conference.org/2027/cfsp.html

Facts from the call for papers (read 2026-10-06):

- Submission deadline: **Nov 17, 2026**, 23:59 AoE. Notification Dec 18, 2026. Conference Feb 15--19, 2027, in person
  only; at least one author must register and attend.
- At most **4 pages**, excluding references and the ethical-considerations section.
- ACM `sigconf` format: `\documentclass[sigconf,anonymous,review]{acmart}`. English, anonymised PDF.
- arXiv preprints are allowed, but the submission must not explicitly associate itself with the posted preprint.
- No concurrent submissions to other venues.
- Relevant topics: "Retrieval, indexing, and ranking with foundation models"; "Search benchmarking and evaluation".

Consequences:

- The paper moves from LNCS to ACM `sigconf` (anonymised for the WSDM version) -- roadmap step 5.
- The 4 pages hold the two research questions of `docs/roadmap.md`; Phase 2 (control signal, fine-tuning,
  BRIGHT) goes to the arXiv extension or a follow-up.

## arXiv date

Decided by the owner on 2026-10-06 (open question 2): the arXiv preprint goes up **before Nov 17, 2026**, with or
just before the WSDM submission. Its content is therefore limited to what is ready in
six weeks; the WSDM version must not refer to it explicitly.
