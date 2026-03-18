# RP1B_Assignment
## Classification task using Machine Learning 
### Overview 
This project aims to classify MGEs plasmids and viruses  and chromosomes using machine learning algorithms trained on sequence based features. 

### Data
The dataset consists of a training folder and a testing folder. Each folder consists of:
- FASTA files (sequences of each element)
  
  Train dataset:
  
  - Chromosomes (n=5000)
  - Plasmids (n=1000)
  - Viruses (n=4000)

  Test dataset:
  - Chromosomes (n=1250)
  - Plasmids (n=250)
  - Viruses (n=1000)
- CSV embeddings

Files are organised in the data/ directory 

### Method

  1.FASTA files are read and parsed using Biopython 

2. Feature extraction k-mer length=5
 
3. Models are trained on the k-mer features and embeddings files
 
4. Models are developed and tuning hyperparameters
 
5. Model performance is evaluated using 5-fold CV and analysing standard classification metrics: F1-score, precision/recall ect.
 
6. Final model evaluation using classification reports and confusion matrices
  
7. Dimensionality reduction

### How to run 

1. Install dependencies in command prompt: pip install biopython pandas numpy scikit-learn matplotlib umap-learn
  
2. Run Final_classifier_.py

3. Classification tool geNomad was used as a benchmark comparison











