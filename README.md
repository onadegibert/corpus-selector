# Corpus Selector
## Project description
The goal of the Corpus Selector is to be able to select from a big available corpus based on domain similarity as compared to a 
smaller corpus in another language. It takes as an input sentences of a specific domain in one language and outputs 
similar sentences in another language. It is based on the LASER sentence embeddings and uses FAISS for similarity.

## Install and run
For creating the virtual environment and installing the dependencies (from requirements.txt), run:

`bash setup.sh`

With the virtual environment activated (source venv/bin/activate), run the following with the python interpreter:

* `(venv) $ python generate_embeddings.py ca en` if you want to generate the embeddings
* `(venv) $ python compute_similarity.py ca en` if you want to compute similarity scores

Substitute the two arguments by the two letter codes of the languages you want to embed. First the low-resourced 
language and secondly, the language with the big corpus.

## Data
The scripts expect two files in the data folder containing one sentence per line that are your corpora. One in the low-resourced language
and one that is the big corpus (eg. ca.txt and en.txt). Two files have been uploaded as reference.