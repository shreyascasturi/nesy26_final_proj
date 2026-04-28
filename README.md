# NeSy Final Project -- Shreyas Casturi -- CS 7820

This is the repo for the NeSy Final project.

The aim of this project was originally to study the effects that reducing LLM size/parameters had on errors within
LLM-generated Lean proofs of math theorems/problems.

Due to significant time constraints, this deliverable was scaled back, to cover:

- Ingesting and scrubbing a dataset for fine-tuning an LLM to generate formal statements + proofs (in Lean) of mathematical theorems/questions
- Running Supervised Fine Tuning with this dataset on a specific model (SmolLM2-135m)
- Running inference on this modified model using the miniF2F benchmark
- Doing spot checks of what the generated proofs contain (do they contain errors? if so, what do they contain?)

# Setup
- We assume that you have basically no resources on your computer (no GPU, etc...).
- You need to setup Ollama on your system. Consult these documents: 
- You need to setup Ollama on your system