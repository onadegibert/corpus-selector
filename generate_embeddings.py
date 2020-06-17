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
    #Check if file exists, UNCOMMENT this lines if there exist already some embeddings you want to use
    # if file_exists(lang) == True:
    #     file = np.loadtxt('embeddings/' + lang + '_embeddings.txt')
    #     embeddings = file
    # else:
    # INDENT the following two lines if you want to use the condition file_exists()
    embeddings = embed(sentences,lang)
    save_embeddings(embeddings,lang)
    return(embeddings)


def main():
    start_time = time.time()


    #Obtain languages from command line
    domain_lang = sys.argv[1]
    target_lang = sys.argv[2]


    # Open files
    domain_sentences = read_file(domain_lang)
    target_sentences = read_file(target_lang)


    # Obtain embeddings if file doesn't exist already
    create_embeddings(domain_sentences,domain_lang)
    create_embeddings(target_sentences,target_lang)


    print("Total elapsed time:")
    print(time.time() - start_time)

main()
