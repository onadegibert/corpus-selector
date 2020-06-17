import numpy as np
import faiss
import time
import datetime
import sys
import re
import csv


def read_file(lang):
    file = open('data/' + lang + '.txt', 'r')
    sentences = file.readlines()
    return (sentences)


def open_embeddings(lang):
    file = np.loadtxt('embeddings/' + lang + '_embeddings.txt')
    embeddings = file
    return(embeddings)


def create_index(target_embeddings):
    d = 1024
    db = target_embeddings.astype('float32')
    index = faiss.IndexFlatL2(d)
    index.add(db)  # add vectors to the index
    return (index)


def get_results(k,query,index):
    D, I = index.search(query, k)     # actual search, returns indices and matrices
    return (I)


def save_results(results,domain_sentences,target_sentences,target_lang):
    date_time = re.sub(r'[-:]', '', datetime.datetime.now().replace(microsecond=0).isoformat())
    output = open('output/' + target_lang + '_similar_sentences_' + date_time + '.csv', 'w', newline='')
    output_writer = csv.writer(output, delimiter=';')

    count = 0
    for sentence in results:
        similar_sentences = [domain_sentences[count].strip()]
        count += 1
        for index in sentence:
            similar_sentences.append(target_sentences[index].strip())
        output_writer.writerow(similar_sentences)

def main():
    start_time = time.time()

    # Obtain languages from command line
    domain_lang = sys.argv[1]
    target_lang = sys.argv[2]

    # Open embeddings files
    domain_embeddings = open_embeddings(domain_lang)
    target_embeddings = open_embeddings(target_lang)

    # Create index
    index = create_index(target_embeddings)

    # Run the query
    query = domain_embeddings.astype('float32')
    k = 5
    results = get_results(k,query,index)

    # Open sentences files
    domain_sentences = read_file(domain_lang)
    target_sentences = read_file(target_lang)


    # Write results to a file
    save_results(results,domain_sentences,target_sentences,target_lang)


    print("Total elapsed time:")
    print(time.time() - start_time)


main()