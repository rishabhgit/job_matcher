# Notes from Agent Pipeline Development for Presentation

## Key Project Goals
- Build a solution to assess the fit of a resume against a job description using AI agents.
- Use a test-driven approach to measure improvements from different experiments.
- Design solution to address common issues with LLMs like hallucination and keep cost low by using smaller models.
- Design a solution that is easy-to-debug and maintain.
- Reduce bias in the solution against protected attributes like age, race, gender and other protected attributes.

## Overall Solution Approach
- Utilize my local setup of Ollama with Llama3, LangChain, PGVector DB and LangFuse
- Find a good Validation Dataset to test various improvement approaches.
    - Arxiv paper search and Kaggle dataset search didn't yield useful result. Kaggle has some dataset on job description but most CVs are synthetic.
- Used a Kaggle dataset on Job descriptions scraped from LinkedIn and selected 5 JDs. Then used Latest Claude Model to generate 6 synthetic CVs per JD with different match levels - Best, average and poor.
- Planned a solution to use Vector DB for storing Resume chunks. Then for each JD, get top CVs from the VectorDB. Lastly use a scoring agent to match JD with top CV and select top 3 CVs for human assessment.

## High Level Solution Design
 - Insert Block diagram here showing the flow of data through various stages in pipeline.

## Generate Test Data
- Used a powerful frontier model to generate synthetic CVs and substitute for SME inputs.

## Data Preparation and Embedding
- Generating Synthetic CVs in Markdown format reduced the need for building Word Dodge or PDF conversion process. 
- Anonymised CVs to remove personal attributes so that the Evaluator agent does not produce a biased evaluation.
- Used Sentence Transformer model to embed and save the resume chunks into VectorDB. Added section headings for more context and better retrieval.
- Built 2 retrieval methods - Reranking using a Cross-encoder and Hybrid search using Recoprical Rank Fusion. Sample tested some records to determine that re-ranking works better. 

## JD and CV Embedding Structured Output generation
- Summarised JD and top CV into structured JSON objects to reduce hallucination from Scoring agent
- However structured JD may miss some synonyms and degrade vector DB search results. At the same time, using the whole JD as search string may exceed the context window of sentence transformer mode.
- To ovecome this issue, stripped away non essential JD information like About company and saved essential JD info along with structured fields.
- The essential JD string can be used to query vector DB and the structured JD fields can be used for fianl scoring of CVs. 

## JD and CV Evaluation
- Built an evaluator agent to match structured JD requirements against structured CV information and provide a match score and recommendation.
- Built an end-to-end pipeline script to find top 10 CVs for each job description, compare the Evaluator agent's score against the ground truth and prepare Spearman Rank correlation score.
- Used several iterations of running pipeline -> reviewing steps in LangFuse -> testing samples in a Notebook to make vairous changes
- Improvement after round 1:
    - Changed the Evaluator prompt from strict skill match to considering semantic equivalent terms - Spearman Rank correlation score improved from 0.34 to 0.4 but median score for Average and Strong match were the same (85).
- Improvement after round 2:
    - Changed the Evaluator prompt again to define score buckets based on parameters - Spearman Rank correlation score fell back to 0.34 with no improvement in score separation between Average and Strong match CVs
- Improvement after round 3:
    - Implemented Chain of Thought by moving the Match_score to the bottom of the class instead of top of the class so that the model matches skills first, prepares justification and then assigns score - Spearman Rank correlation score improved to 0.40 with no improvement in separation between Average and  Strong match CVs median score (70).
- Improvement after round 4:
    - Fixed a bug in the scoring pipeline. If a CV from a different role category was being evaluated, downgraded its ground truth to Bad Match (E.g. a Strong Match Sr. Data Scientist role when evaluated for the JD of AI Research Scientist was downgraded to Bad Match). Spearman Rank correlation score improved to 0.5 and the separation between Strong and Average match CVs median score improved to 13 points (83 vs 70)

## Project Outcome:
- The current solution uses a combination of Sentence Transformer and LLMs to automate the process of evaluation and short-listing candidates for a role.
- Modular solution design and use of LangFuse have enabled test driven improvement to each step of the pipeline.
- Bias is minimised by removing personal data from resumes and anonymizing the cleaned resume before embedding it using Sentence Transformer.
- Hallucination is minimised by distilling free text data into structured json fields whereever feasible.
- The solution utilises small, cheaper LLMs to achieve a reasonable level of performance.

## Improvement Areas
- Build a robust data preprocessing pipeline to ingest CVs various formats like docx, pdf etc.
- Build a user interface and an interactive chatbot for online CV matching and refinement based on user feedback.
- For batch processing, build a feedback loop to assess final labels from humans, identify drift and improve the solution accordingly.
- Build a bigger and more diverse dataset of resumes and labels to evaluate approaches like ReAct can improve performance, especially on borderline cases.