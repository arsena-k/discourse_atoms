# discourse_atoms

*GitHub repository to accompany research paper in preparation "Discourses of Death: Extracting the Semantic Structure of Lethal Violence with Machine Learning," by Alina Arseniev-Koehler, Susan Cochran, Vickie Mays, Kai-Wei Chang, and Jacob G. Foster.  Preprint available at: https://osf.io/preprints/socarxiv/nkyaq/ Please cite this repository or paper if reused. Code written in Python 3 in Windows.* 

**Paper Abstract:** Violent deaths — such as homicides and suicides — can be a source of "senseless suffering.'' Forensic authorities use narratives to trace events surrounding these deaths both in a search for meaning and a desire to prevent them. We examine these narratives, drawing on summaries from over 300,000 violent deaths in the U.S. Our goal is to capture the latent themes and other semantic structures that undergird violent death and its surveillance. To do so, we introduce a flexible model for the analysis of topics ("discourse atoms''), where locations in a semantic embedding space map onto distributions over words. We use this model to extract a high-level, thematic description of discourses of violent death. We identify 225 topics, discussing several in detail including topics about drug paraphernalia, financial difficulties, erratic and impaired behavior, and uncertainty. Our results offer clues into themes surrounding violent death and its administrative surveillance, and our method offers a new way to computationally model discourse.

The Discousre Atom Topic Model builds directly on a generative model for word embeddings themselves, proposed by Sanjeev Arora and colleagues:
* Arora, Sanjeev, et al. "A latent variable model approach to pmi-based word embeddings." Transactions of the Association for Computational Linguistics 4 (2016): 385-399.
* Arora, Sanjeev, Yingyu Liang, and Tengyu Ma. "A simple but tough-to-beat baseline for sentence embeddings." (2016).
* Arora, Sanjeev, et al. "Linear algebraic structure of word senses, with applications to polysemy." Transactions of the Association for Computational Linguistics 6 (2018): 483-495.

**Code in this repository (in develoment) will show our methods** to implement the discourse atom topic model. 
