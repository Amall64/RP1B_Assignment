#!/usr/bin/env python
# coding: utf-8

# In[131]:


#Load the FASTA files +store the sequences +assign a label for each class 


# In[304]:






# In[305]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
#set up 


# In[306]:


#defining k-mer extraction code 

#k-mer extraction from the sequences 

#use DictVectorizer or CountVectorizer


# In[307]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#evaluate a model 
#randomly shuffles the data and splits it into 5
#each part each fold keeps the same class proportions as the original training data


# In[308]:


#we want to load the sequences and label each sequence 
from Bio import SeqIO   #We want to read and parse the files using biopython module 
from sklearn.feature_extraction.text import CountVectorizer  #countvectorizer will convert seq into numerical vectors 

sequences = []
labels = []
ids = []  #for saving the ids of each fasta 


for record in SeqIO.parse(
    "data/train/genomad_benchmark_chromosome_n5000.fna", 
    "fasta"
):
    sequences.append(str(record.seq))
    labels.append("chromosome")
    ids.append(record.id)

#the plasmid file 
for record in SeqIO.parse(  #Reads each file 
    "data/train/genomad_benchmark_plasmid_n1000.fna",  #file name 
    "fasta"
):
    sequences.append(str(record.seq)) #Converts to string
    labels.append("plasmid")  #links the label plasmid to the chosen FASTA file
    ids.append(record.id)  #saves fasta id header to ids list 

#example how this should look:    
#sequences = ["ATGCATCGATCG"]
#labels = ["plasmid"]    
    
    
for record in SeqIO.parse(
    "data/train/genomad_benchmark_virus_n4000.fna",
    "fasta"
):
    sequences.append(str(record.seq))
    labels.append("virus")
    ids.append(record.id)
    
print(len(sequences)) #1000 plasmids, 5000 chromosomes and 4000 viruses = 10,000
print(len(labels)) #should also be 10,000


# In[309]:


#load the embedding csvs 

import pandas as pd  #reading csv files 
import numpy as np

chrom_pt  = pd.read_csv("data/train/emb_ntv2_500m_chromosome_n5000_embeddings.csv").drop(columns=["Header"])  #removes the string 'Header' so the model can actually work (only on numbers)
plasm_pt  = pd.read_csv("data/train/emb_ntv2_500m_plasmid_n1000_embeddings.csv").drop(columns=["Header"])
virus_pt  = pd.read_csv("data/train/emb_ntv2_500m_virus_n4000_embeddings.csv").drop(columns=["Header"])

chrom_ft  = pd.read_csv("data/train/emb_ntv2_500m_ft_chromosome_n5000_embeddings.csv").drop(columns=["Header"])
plasm_ft  = pd.read_csv("data/train/emb_ntv2_500m_ft_plasmid_n1000_embeddings.csv").drop(columns=["Header"])
virus_ft  = pd.read_csv("data/train/emb_ntv2_500m_ft_virus_n4000_embeddings.csv").drop(columns=["Header"])

#now we have 6 dataframes but need to combine them 


 
x_pretrained = pd.concat([chrom_pt, plasm_pt, virus_pt])  #combine pretrained dataframes 

x_finetuned = pd.concat([chrom_ft, plasm_ft, virus_ft])  #combine finetuned dataframes 



labels = np.array(   #builds the label array 
    ["chromosome"] * len(chrom_pt) +  #5000 chromosome labels listed 
    ["plasmid"] * len(plasm_pt) +  #using pt as example but the no. of labels is the same 
    ["virus"] * len(virus_pt)
)
#defining x_pretrained and x_finetuned 
x_pretrained = x_pretrained.to_numpy()  #converts from pandas to numpy arrays 
x_finetuned = x_finetuned.to_numpy()

y = labels
#make sure the order is chrom,plas,virus do y=labels can be used again 


# #just to make sure there are no strings in the input for the model 
# print(chrom_pt.columns[:5])  # see the first 5 column names
# print(chrom_pt.head(2))      # see first 2 rows

# In[310]:


#extract k-mers from all sequences using countvectorizer

ksize = 5 

def sequence_to_kmers(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)] #slides 5-mer sized window along DNA seq
    return ' '.join(kmers) #space separation for the countvectoriser 

#converts all the sequences to k-mer sized strings
print(f"\nExtracting {ksize}-mers from all sequences...")
kmer_strings = [sequence_to_kmers(seq, ksize) for seq in sequences] #converts raw DNA into kmer 


    
   


# In[311]:


#train:test split for pt and ft
from sklearn.model_selection import train_test_split

x_train_pt, x_test_pt, y_train, y_test = train_test_split(
    x_pretrained,  #pt embeddings feature 
    y,  #class label e.g. chrom
    test_size=0.2,  #keeps 20% for testing and 80% for training 
    stratify=y,   #makes sure class distribution is balanced between the splits 
    random_state=42  #makes the split reproducible 
)


x_train_ft, x_test_ft, y_train, y_test = train_test_split(
    x_finetuned,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#_train for this stage _test not to touch until testing stage 


# In[312]:


#train-test split for kmers 
indices = np.arange(len(sequences))  #creates a list for each number of sequences 
train_idx, test_idx = train_test_split(  #splits the numbers between train (80%) and val (20%)
    indices, test_size=0.2, stratify=y, random_state=42
)

vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b') #takes each sequence and treats each piece indivdually
x_train_kmer = vectorizer.fit_transform([kmer_strings[i] for i in train_idx]).toarray()
x_test_kmer_val = vectorizer.transform([kmer_strings[i] for i in test_idx]).toarray()

x_kmers = np.vstack([x_train_kmer, x_test_kmer_val])

#convert to feature matrix
#show some feature names (k-mers)
feature_names = vectorizer.get_feature_names_out()  #gives the names of the k-mers as they appear in the matrix 


# In[316]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

baseline_model = RandomForestClassifier(
    n_estimators=100,  #100 trees
    max_depth=None,  #trees depth dont have a limit 
  #  class_weight ='balanced', not used for this 
    random_state=42,
    n_jobs=-1  #uses all CPU to speed up computation
)

# K-mers
baseline_model.fit(x_train_kmer, y_train) #trains this model using kmer features
y_pred_kmer_base = baseline_model.predict(x_test_kmer_val)  #stores the prediction
print("K-mers Baseline:")
print(classification_report(y_test, y_pred_kmer_base, target_names=["chromosome", "plasmid", "virus"]))  #compares true labels vs predicted 

# Pretrained
baseline_model.fit(x_train_pt, y_train)
y_pred_pt_base = baseline_model.predict(x_test_pt)
print("Pretrained Baseline:")
print(classification_report(y_test, y_pred_pt_base, target_names=["chromosome", "plasmid", "virus"]))

# Fine-tuned
baseline_model.fit(x_train_ft, y_train)
y_pred_ft_base = baseline_model.predict(x_test_ft)
print("Fine-tuned Baseline:")
print(classification_report(y_test, y_pred_ft_base, target_names=["chromosome", "plasmid", "virus"]))


# In[317]:


#runs CV and returns metrics (how well the model performed)
#x = features y = labels model = RF 
def run_cv(X, y, model):
    results = cross_validate(model, X, y, cv=skf,  #runs stratified fold 
                             scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
    acc       = results["test_accuracy"]
    f1        = results["test_f1_macro"]
    precision = results["test_precision_macro"]
    recall    = results["test_recall_macro"]
    
    return (acc.mean(), acc.std(), 
            f1.mean(), f1.std(), 
            precision.mean(), precision.std(), 
            recall.mean(), recall.std())

#acc.mean = av accuracy across 5 folds  acc.std = how much has accuracy varied 


# In[320]:


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def evaluate_model(X, y, name, n_splits=5):  #5-fold CV used to tune hyperparameters 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  #makes sure each fold has the same class distribution as the full dataset 
    
    X = np.array(X) #converts inputs into numpy arrays so it works with sckitlearn 
    y = np.array(y)

    model = RandomForestClassifier( #metrics decided on after testing different parameters 
        n_estimators=200,
        max_depth= 20,
        class_weight = 'balanced',
        random_state=42,
        n_jobs=-1
    )
    
    
    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    
     
    y_all_pred = np.zeros_like(y) #stores all predictions
    
    for train_idx, val_idx in skf.split(X, y):  #the loop iterate through each fold (5)
        X_train, X_val = X[train_idx], X[val_idx]  #splits features into train/val
        y_train, y_val = y[train_idx], y[val_idx]  
        
        model.fit(X_train, y_train)  #trains model rf on the fold its working on 
        y_pred = model.predict(X_val) #predictions are tested on the held out test 
        
        y_all_pred[val_idx] = y_pred  
        
        accuracy_list.append(accuracy_score(y_val, y_pred))
        f1_list.append(f1_score(y_val, y_pred, average='macro'))
        precision_list.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        recall_list.append(recall_score(y_val, y_pred, average='macro'))
    
    #all metrices 
    print(f"\n{name}")
    print("Accuracy_mean:", np.mean(accuracy_list))
    print("Accuracy_std:", np.std(accuracy_list))
    print("MacroF1_mean:", np.mean(f1_list))
    print("MacroF1_std:", np.std(f1_list))
    print("Precision_mean:", np.mean(precision_list))
    print("Precision_std:", np.std(precision_list))
    print("Recall_mean:", np.mean(recall_list))
    print("Recall_std:", np.std(recall_list))
    
#only runs the cv on the 8,000 training sequences


# In[322]:


evaluate_model(x_train_kmer, y_train, "K-mers")

evaluate_model(x_train_pt,   y_train, "Pretrained")

evaluate_model(x_train_ft,   y_train, "Fine-tuned")



# In[323]:


#after testing different parameters, run the classifier on x_test 
#evaluate_test trains once, predicts using x_test 
from sklearn.metrics import classification_report

def evaluate_test(X_train, X_test, y_train, y_test, name):  #no split this time 
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)  #trains on full x_train set and predicts on x_test
    
    y_pred = model.predict(X_test)  #uses the trained models to make predictions 
    
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["chromosome", "plasmid", "virus"]))


    return y_pred
#running on remaining 2000 samples 




# In[324]:


evaluate_test(x_train_kmer, x_test_kmer_val, y_train, y_test, "K-mers")  

evaluate_test(x_train_pt,   x_test_pt,   y_train, y_test, "Pretrained")

evaluate_test(x_train_ft,   x_test_ft,   y_train, y_test, "Fine-tuned")


# In[325]:


#now final test with independent test data 
#1 once uploaded read and parse each file 

from Bio import SeqIO   #We want to read and parse the files using biopython module 
from sklearn.feature_extraction.text import CountVectorizer  #countvectorizer will convert seq into numerical vectors 

test_sequences = []
test_labels = []
test_ids = []  #for saving the ids of each fasta this time into test dictionaries 


for record in SeqIO.parse(
    "data/test/test_chromosome_n1250.fna",
    "fasta"
):
    test_sequences.append(str(record.seq))
    test_labels.append("chromosome")
    test_ids.append(record.id)

#the plasmid file 
for record in SeqIO.parse(  #Reads each file 
    "data/test/test_plasmid_n250.fna",  #file name 
    "fasta"
):
    test_sequences.append(str(record.seq)) #Converts to string
    test_labels.append("plasmid")  #links the label plasmid to the chosen FASTA file
    test_ids.append(record.id)  #saves fasta id header to ids list     
    
    
for record in SeqIO.parse(
    "data/test/test_virus_n1000.fna",
    "fasta"
):
    test_sequences.append(str(record.seq))
    test_labels.append("virus")
    test_ids.append(record.id)
    
print(len(test_sequences)) #250 plasmids, 1250 chromosomes and 1000 viruses = 2500
print(len(test_labels)) #should also be 2500


# In[326]:


#load embeddings csv 


import pandas as pd  #reading csv files 
import numpy as np

test_chrom_pt = pd.read_csv("data/test/test_ntv2_500m_chromosome_n1250_embeddings.csv").drop(columns=["Header"]) #removes header to leaving only embeddings 
test_plasm_pt = pd.read_csv("data/test/test_ntv2_500m_plasmid_n250_embeddings.csv").drop(columns=["Header"])
test_virus_pt = pd.read_csv("data/test/test_ntv2_500m_virus_n1000_embeddings.csv").drop(columns=["Header"])

test_chrom_ft = pd.read_csv("data/test/test_ntv2_500m_ft_chromosome_n1250_embeddings.csv").drop(columns=["Header"])
test_plasm_ft = pd.read_csv("data/test/test_ntv2_500m_ft_plasmid_n250_embeddings.csv").drop(columns=["Header"])
test_virus_ft = pd.read_csv("data/test/test_ntv2_500m_ft_virus_n1000_embeddings.csv").drop(columns=["Header"])

#now we have 6 dataframes but need to combine them 


 
x_test_pretrained = pd.concat([test_chrom_pt, test_plasm_pt, test_virus_pt])  #combine pretrained dataframes 

x_test_finetuned = pd.concat([test_chrom_ft, test_plasm_ft, test_virus_ft])  #combine finetuned dataframes 



test_labels = np.array(   #builds the label array 
    ["chromosome"] * len(test_chrom_pt) +  #5000 chromosome labels listed 
    ["plasmid"] * len(test_plasm_pt) +
    ["virus"] * len(test_virus_pt)
)
#defining x_pretrained and x_finetuned 
x_test_pretrained = x_test_pretrained.to_numpy()  #converts from pandas to numpy arrays 
x_test_finetuned = x_test_finetuned.to_numpy()

y_test_ind = test_labels


# In[327]:


# using the extract kmers code again 

test_kmer_strings = [sequence_to_kmers(seq, ksize) for seq in test_sequences]
x_test_kmer = vectorizer.transform(test_kmer_strings).toarray()


# In[328]:


y_pred_kmer = evaluate_test(x_train_kmer, x_test_kmer,       y_train, y_test_ind, "K-mers - Independent Test")

y_pred_pt   = evaluate_test(x_train_pt,   x_test_pretrained, y_train, y_test_ind, "Pretrained - Independent Test")

y_pred_ft   = evaluate_test(x_train_ft,   x_test_finetuned,  y_train, y_test_ind, "Fine-tuned - Independent Test")

cm_kmer = confusion_matrix(y_test_ind, y_pred_kmer)
print("K-mers Independent Test Confusion Matrix:")
print(cm_kmer)

cm_pt = confusion_matrix(y_test_ind, y_pred_pt)
print("\nPretrained Independent Test Confusion Matrix:")
print(cm_pt)

cm_ft = confusion_matrix(y_test_ind, y_pred_ft)
print("\nFine-tuned Independent Test Confusion Matrix:")
print(cm_ft)


# In[329]:


#kmer PCA and UMAP 

import numpy as np #for numerical 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt #to see the PCA results as a scatterplot
import umap



X_dense = np.vstack([x_train_kmer, x_test_kmer_val])  #PCA/UMAP needs it to be a regular dense matrix not sparse 

#PCA
print("Running PCA")
pca_pre_umap = PCA(n_components=50, random_state=42)
X_reduced = pca_pre_umap.fit_transform(X_dense)  #fits the PCA to the kmers data 
print(f"  Variance explained: PC1={pca_pre_umap.explained_variance_ratio_[0]:.3f}  "
      f"PC2={pca_pre_umap.explained_variance_ratio_[1]:.3f}")

X_pca_kmers = X_reduced[:, :2]

#UMAP 
print("Running UMAP")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    random_state=None,  
    n_jobs=-1           #all CPU cores
)
X_umap_kmers = reducer.fit_transform(X_reduced)

#plotting
colours   = {"chromosome": "steelblue", "plasmid": "darkorange", "virus": "green"}
labels_arr = np.array(labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, coords, title, (xlabel, ylabel) in zip( #links each part of the list to make one subplot 
    axes,
    [X_pca_kmers, X_umap_kmers],
    [f"PCA — k-mer (k={ksize})", f"UMAP — k-mer (k={ksize})"],
    [("PC1", "PC2"), ("UMAP1", "UMAP2")]
):
    for cls in ["chromosome", "plasmid", "virus"]:
        mask = labels_arr == cls  #selects samples belonging to that class
        ax.scatter(coords[mask, 0], coords[mask, 1], #plots samples 
                   label=cls, alpha=0.4, s=10, c=colours[cls])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=2)

plt.tight_layout()
plt.savefig(f"kmer_k{ksize}_pca_umap.png", dpi=150)
plt.show()







# In[330]:


#PCA of PT and FT 
# k-mer PCA and UMAP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


X_dense = np.vstack([x_train_kmer, x_test_kmer_val]) #combines train + test k-mer arrays into dense


print("Running PCA")
pca_pre_umap = PCA(n_components=50, random_state=42)
X_reduced = pca_pre_umap.fit_transform(X_dense)
print(f"  Variance explained: PC1={pca_pre_umap.explained_variance_ratio_[0]:.3f}  "
      f"PC2={pca_pre_umap.explained_variance_ratio_[1]:.3f}")


X_pca_kmers = X_reduced[:, :2] #2 PC for plots


print("Running UMAP")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    random_state=42,  
    n_jobs=-1
)
X_umap_kmers = reducer.fit_transform(X_reduced)


colours = {"chromosome": "steelblue", "plasmid": "darkorange", "virus": "green"}
labels_arr = np.array([labels[i] for i in train_idx] + [labels[i] for i in test_idx])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, coords, title, (xlabel, ylabel) in zip(
    axes,
    [X_pca_kmers, X_umap_kmers],
    [f"PCA — k-mer (k={ksize})", f"UMAP — k-mer (k={ksize})"],
    [("PC1", "PC2"), ("UMAP1", "UMAP2")]
):
    for cls in ["chromosome", "plasmid", "virus"]:
        mask = labels_arr == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cls, alpha=0.4, s=10, c=colours[cls])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=2)

plt.tight_layout()
plt.savefig(f"kmer_k{ksize}_pca_umap.png", dpi=150)
plt.show()
