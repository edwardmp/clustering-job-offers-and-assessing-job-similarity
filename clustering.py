import logging 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from gensim import corpora, models, similarities, matutils
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.pipeline import Pipeline
from math import log
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import jaccard_similarity_score

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loadStopWords():
    stopWords = [line.rstrip('\n\r') for line in open('stop-words_dutch.txt')]
    stopWords = set(stopWords) # only unique elements
    print "Number of Dutch Stop Words loaded: %d\n" % len(stopWords)
    return stopWords

def openInputDataFileAndReturnSentencesRows(fileName='input/data.csv'):
    f = open(fileName)
    csv_f = csv.reader(f)

    jobs = []
    jobIds = []
    index = 0
    previousJobId = 0
    for line in csv_f:
		if line[1] != "sentence" or line[0] != "job_id":
			jobIds.append(line[0])
			jobs.append(line[1])

    return (jobs, jobIds)

def writeClustersForEachJobIdToOutputFile(clusterLabels, allJobIds, sentenceList, fileName="output/outputClusterForJobId.csv"):
    with open(fileName, "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["JobId", "Cluster", "Sentence"])
     
        count = 0
        for sentence in sentenceList:
            clusterNumber = clusterLabels[count]
            writer.writerow([jobIds[count], clusterNumber, sentence])
            count += 1

def writeInfoForEachJobIdComboToOutputFile(jobCombinationAndCoefficients, fileName="output/outputInfoForJobIdCombo.csv"):
    with open(fileName, "w+") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["JobId_First", "JobId_Second", "Jaccard_Similarity_Score"])
     
        for job in jobCombinationAndCoefficients:
            writer.writerow([job["job_id_first"], job["job_id_second"], job["jaccard_similarity_score"]])
            
def calculateNumberOfIdealClusters(maxAmount, corpus):
    range_n_clusters = range(2, maxAmount) # max amount of clusters equal to amount of jobs

    silhouette_high = 0;
    silhouette_high_n_clusters = 2;

    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity="euclidean")
        cluster_labels = cluster.fit_predict(corpus)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(corpus, cluster_labels)

        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        if (silhouette_avg > silhouette_high):
            silhouette_high = silhouette_avg
            silhouette_high_n_clusters = n_clusters

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(corpus, cluster_labels)

    print ("Highest score = %f for n_clusters = %d" % (silhouette_high, silhouette_high_n_clusters))
    return silhouette_high_n_clusters


def calculate_amount_of_clusters_for_each_job_id(clusterLabels, jobIds):
    dictionaryWithClustersForEachJobId = {}

    for index, value in enumerate(clusterLabels):
         jobIdForSentence = jobIds[index]

         if jobIdForSentence not in dictionaryWithClustersForEachJobId:
            dictionaryWithClustersForEachJobId[jobIdForSentence] = {}

         jobIdDict = dictionaryWithClustersForEachJobId[jobIdForSentence]

         if value not in jobIdDict:
            num = 0
         else:
            num = jobIdDict[value]

         jobIdDict[value] = num + 1

    return dictionaryWithClustersForEachJobId

def calculateJaccardScoreForEachJobCombo(allJobIds, clustersForEachjobSentence, clusterLabels):
    jobCombinationAndCoefficients = []
    index = 0;
    for jobId in list(set(jobIds)):
        copy_list = list(set(jobIds[:])) # only unique jobIds, after that convert back to list
        del copy_list[index] #remove self

        for eachOtherJobId in copy_list:
            otherJobDict = clustersForEachjobSentence[eachOtherJobId]
            currentJobDict = clustersForEachjobSentence[jobId]

            for key in set(clusterLabels):
                if key not in currentJobDict:
                    currentJobDict[key] = 0

                if key not in otherJobDict:
                    otherJobDict[key] = 0

            coefficient = jaccard_similarity_score(currentJobDict.values(), otherJobDict.values())
            jobCombinationAndCoefficients.append({"job_id_first": jobId, "job_id_second": eachOtherJobId, "jaccard_similarity_score": coefficient})

            if (coefficient >= 0.00):
                print "Job with ids %s and %s are %f similar %s %s" % (jobId, eachOtherJobId, coefficient, currentJobDict, otherJobDict)

        index += 1

    return jobCombinationAndCoefficients

sentenceList, jobIds = openInputDataFileAndReturnSentencesRows()

# Automatically detect common phrases, bigram words are underscored in between them (frequently co-occurring tokens)
bigram = models.Phrases([line.lower().split() for line in sentenceList])

# Load a list of stopwords
stopWords = loadStopWords()

# convert tokens to vector
bigramAsDictionary = corpora.Dictionary(bigram[[line.split() for line in sentenceList]])

# convert stopwords to id
stopWordIds = [bigramAsDictionary.token2id[stopword] for stopword in stopWords
          if stopword in bigramAsDictionary.token2id]

# remove stopwords
bigramAsDictionary.filter_tokens(stopWordIds)
bigramAsDictionary.compactify()

# convert to (word_id, word_frequency) tuples
bagOfWords = [bigramAsDictionary.doc2bow(bigram[line.split()]) for line in sentenceList]

# convert matrix to term frequencies matrix
tfidf = models.TfidfModel(bagOfWords)
corpusTfidf = tfidf[bagOfWords]

# use LSA analysis on corpus
lsi = models.LsiModel(corpusTfidf, id2word=bigramAsDictionary, num_topics=30)
corpusLsi = lsi[corpusTfidf]

# convert corpus to array
corpusAsMatrix = matutils.corpus2dense(corpusLsi, num_terms=30).transpose()

# calculate ideal number of clusters based on silhouette analysis
numClusters = calculateNumberOfIdealClusters(len(set(jobIds)), corpusAsMatrix)

cluster = AgglomerativeClustering(n_clusters=numClusters, linkage="ward", affinity="euclidean")
clusterLabels = cluster.fit_predict(corpusAsMatrix)

clustersForEachjobSentence = calculate_amount_of_clusters_for_each_job_id(clusterLabels, jobIds)

jobCombinationAndCoefficients = calculateJaccardScoreForEachJobCombo(jobIds, clustersForEachjobSentence, clusterLabels)

writeInfoForEachJobIdComboToOutputFile(jobCombinationAndCoefficients)
writeClustersForEachJobIdToOutputFile(clusterLabels, jobIds, sentenceList)
