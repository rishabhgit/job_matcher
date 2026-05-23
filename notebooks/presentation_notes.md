# Notes from Agent Pipeline Development for Presentation

## Overall Solution Approach
- Utilize my local setup of Ollama with Llama3, LangChain, PGVector DB and LangFuse
- Find a good Validation Dataset to test various improvement approaches.
    - Extensive Google search didn't yield useful result. Kaggle has some dataset on job description but most CVs are synthetic.
- Used a Kaggle dataset on Job descriptions scraped from LinkedIn and selected 5 JDs. Then used Latest Claude Model to generate 6 synthetic CVs per JD with different match levels - Best, average and poor.
- Planned a solution to use Vector DB for storing Resume chunks. Then for each JD, get top CVs from the VectorDB. Lastly use a scoring agent to match JD with top CV and select top 3 CVs for human assessment.

## Generate Test Data
- To be completed

## Data Preparation and Embedding

## JD and CV Embedding and Evaluation
- Summarised JD and top CV into structured JSON objects to reduce hallucination from Scoring agent
- However structured JD may miss some synonyms and degrade vector DB search results. At the same time, using the whole JD as search string may exceed the context window of sentence transformer mode.
- To ovecome this issue, stripped away non essential JD information like About company and saved essential JD info along with structured fields.
- The essential JD string can be used to query vector DB and the structured JD fields can be used for fianl scoring of CVs. 
