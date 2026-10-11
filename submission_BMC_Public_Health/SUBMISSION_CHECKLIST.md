# BMC Public Health submission: checklist

Portal: Springer Nature submission system, reached through the "Submit manuscript" link on the BMC Public Health journal page. Article type: **Research**.

## Blocking issues (fix these before uploading)

- [ ] **Ethics approval.** The manuscript currently says "No separate research ethics committee review was undertaken" and that approval came from Davis Company. BMC requires a named ethics committee/IRB that either **approved** the study or formally **waived** approval and consent, with a reference number. Without one, the manuscript will very likely be rejected at the desk. Get a retrospective approval or waiver (for example, from a university or medical-science ethics committee such as one registered with IR.NREC) and rewrite the statement. Example wording: "The study was approved by [Committee], which waived the requirement for informed consent (ref. [ID])."
- [x] **Corresponding author.** Masih Mousavi (m451hm@gmail.com) is now on the title page and in the cover letter.
- [x] **Code availability.** It now says the code is available at https://github.com/m451h/ESAM. Optional: archive a release on Zenodo to get a DOI.

## Recommended fixes

- [x] Affiliations should list institutions, not job titles. "Data Scientist and CTO, Davis Company" should become "Davis Company, Tehran, Iran" (done: they are merged into one affiliation, ², and the rest are renumbered).
- [x] Author line: add the missing space in "Mousavi³⁴,Reza".
- [x] Remove the bullet points before "Authors:" and "Affiliations:" on the title page.
- [ ] Figures: also upload Figure 1 and Figure 2 as separate files (`figures/`, 300 dpi PNG). Keep the figure legends in the manuscript text.
- [ ] Rename `Manuscript.docx` if you want to, but keep the line numbering (added in this copy).
- [ ] Generative AI: BMC asks for this in Methods or Acknowledgements. The current "Use of generative AI" declaration is probably acceptable, but the editor may ask you to move it.

## Already OK

- Abstract: 294 words (limit 350), structured as Background/Methods/Results/Conclusions.
- Keywords: 8 (3–10 allowed).
- Declarations section: all required headings are present.
- Vancouver-style numbered references.
- Reporting checklists (TRIPOD+AI, RECORD) included as Additional files 3–4. Upload the checklist when the portal asks for one.
- Page numbers are present. Line numbers were added in `Manuscript.docx`.

## Files to upload

| Portal file type | File |
|---|---|
| Manuscript | `Manuscript.docx` |
| Figure | `figures/Figure1.png`, `figures/Figure2.png` |
| Supplementary material | `additional_files/*` (5 DOCX + 6 CSV) |
| Cover letter | `cover_letter.docx` (fill in the bracketed fields) |

## Other things the portal will ask for

- ORCID iDs for all authors, at least for the submitting author.
- Confirmation that all co-authors agree to the submission.
- Suggested reviewers (optional).
- The article processing charge (APC). Check BMC's current APC and its waiver/discount country list before you submit. If you are eligible, ask for the waiver at submission.
