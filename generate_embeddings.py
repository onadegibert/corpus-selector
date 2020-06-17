import sys
import os.path
from laserembeddings import Laser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import datetime
import time
import re


def read_file(lang):
    file = open('data' + lang + '.txt', 'r')
    sentences = file.readlines()
    return (sentences)


def file_exists(lang):
    if os.path.isfile('embeddings/'+lang+'_embeddings.txt'):
        overwrite = input("There's already a sentence embeddings file for {}. Do you want to overwrite it? (y/n)\n".format(lang))
        return False if overwrite == "y" else True
    else:
        return False


def embed(sentences, lang):
    laser = Laser()
    print("Embedding sentences for {}...".format(lang))
    embeddings = laser.embed_sentences(sentences, lang=lang)
    return embeddings


def save_embeddings(embeddings,lang):
    file = open('embeddings/'+lang+'_embeddings.txt','w')
    file.truncate()
    np.savetxt(file,embeddings)
    file.close()


def create_embeddings(sentences,lang):
    #Check if file exists
    if file_exists(lang) == True:
        file = np.loadtxt('embeddings/' + lang + '_embeddings.txt')
        embeddings = file
    else:
        embeddings = embed(sentences,lang)
        save_embeddings(embeddings,lang)
    return(embeddings)


def write_most_similar(scores,target_language,domain_sentences,target_sentences):
    date_time = re.sub(r'[-:]','',datetime.datetime.now().replace(microsecond=0).isoformat())
    output = open('output/' + target_language+'_similar_sentences_'+date_time+'.csv', 'w', newline='')
    output_writer = csv.writer(output, delimiter=';')
    i = 0
    for row in scores:
        indeces = np.argsort(-row)[:5]  # sorts the row per values and returns the indices of the highest scores
        similar_sentences = [domain_sentences[i].strip()]
        i += 1
        for index in indeces:
            similar_sentences.extend([target_sentences[index].strip()])
        output_writer.writerow(similar_sentences)


def main():
    start_time = time.time()

    #Obtain languages from command line
    domain_lang = sys.argv[1]
    target_lang = sys.argv[2]

    # Open files
    domain_sentences = read_file(domain_lang)
    target_sentences = read_file(target_lang)

    # Obtain embeddings if file doesn't exist already
    domain_embeddings = create_embeddings(domain_sentences,domain_lang)
    target_embeddings = create_embeddings(target_sentences,target_lang)

    # Obtain cosine similarity scores
    #print("Obtaining cosine similarity scores...")
    #scores = cosine_similarity(domain_embeddings,target_embeddings)

    # Write them to a file
    #write_most_similar(scores,target_lang,domain_sentences,target_sentences)
    print("Total elapsed time:")
    print(time.time() - start_time)

main()
