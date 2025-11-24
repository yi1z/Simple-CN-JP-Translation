# WCC-JC Sports Filter v2

## Motivation: Why do we filter the data?

If you want to train a **domain-specialized Japanese–Chinese translation model** (e.g., focused on **sports**), it is usually not enough to train on a generic parallel corpus. Domain-specific data helps the model:

- Learn **domain terminology** (“篮板球”, “三分线”, “延长戦” …)
- Learn **typical sentence patterns** and style (commentary, match reports, analysis)
- Improve **in-domain accuracy** without damaging general performance too much

The goal of this script is to **automatically mine sports-domain sentence pairs** from a large mixed-domain JA–ZH corpus such as WCC-JC, so that you can use this subset to:

- Fine-tune / distill a sports-focused translation model, or  
- Use it as a sports-heavy component of a multi-domain training mixture.

---

## What this script does (high-level)

For given WCC-JC-style corpus (`wccjc_all/`):

1. **Automatically discover JA–ZH parallel data** in the directory (split files and TSV-like files).
2. **Apply basic length-based filtering** to remove noisy or misaligned pairs.
3. **Use sports-related keywords** (in both JA and ZH) to **coarsely recall** candidate pairs that might be about sports.
4. **Train a “sports centroid”** in sentence embedding space:
   - Encode keyword-positive sports pairs using a multilingual `sentence-transformers` model
   - Compute the mean vector ⇨ this is the “sports semantic center”
5. **Scan the whole corpus with this centroid**:
   - For each keyword-positive candidate, encode it and compute cosine similarity with the sports centroid
   - Keep the pair if the similarity is above a configurable `threshold`
6. **Write the final sports subset** as a TSV file:  

   ```text
   <JA sentence>\t<ZH sentence>
