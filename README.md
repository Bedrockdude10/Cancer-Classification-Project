# Cancer-Classification-Project
Code written for the final project for MATH 4570: Matrix Methods in Machine Learning &amp; Data Analysis.

Early detection and diagnosis is critical in the treatment of cancer.
Recently, microarray gene expression data has been used to this end.
There is a wealth of genomic data available for use in diagnosis, the 
challenge now faced by researchers and medical practitioners is extracting 
meaningful information from all this data.

We propose an application of machine learning to address the multi-class classification
problem presented by the microarray data. In particular, we will use deep neural networks
to analyze the curated microarray datasets provided by the Structural Bio-informatics and 
Computational Biology Lab in the hopes of outperforming the benchmarks on these datasets,
and building a more general model which can accurately diagnose a variety of cancers.

After training our model, we will assess it's performance on a dataset including both
healthy and cancerous cells. We will compute the precision and recall of our model(s)
and discuss the implications of the different types of errors in a medical setting.

### Data
Our datasets are from the Curated Microarray Database (https://sbcb.inf.ufrgs.br/cumida)
created by the Structural Bioinformatics and Computational Biology Lab.

We have downloaded a subset of the data provided, and merged files with the same set of features
(genes) to produce a multi-class dataset where the target variable is the type
of cancer present in each sample (there are healthy samples as well).

We generated datasets according to the procedure above for two different sets of genes, GPL96 and GPL570. Our goal is to
analyze the datasets independently and use these results to determine which set of genes
has more predictive power.

Here is link to the data https://drive.google.com/drive/folders/1eJelCkSJAd7yWjFGcTLycNLSvzpQWK_A?usp=sharing.
The csv files contain the processed datasets corresponding to each set of genes. The zip files
contain all the data we download from the Curated Microarray Database to produce these merged datasets.
To run the join and label notebooks, simply download and unzip the zip files.