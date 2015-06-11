import logging 
from gensim import corpora, models, similarities, matutils
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.pipeline import Pipeline
from math import log
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
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

sentenceList, jobIds = openInputDataFileAndReturnSentencesRows()

bigram = models.Phrases([line.lower().split() for line in sentenceList])

stopWords = loadStopWords()

dictionary = corpora.Dictionary(bigram[line.lower().split()] for line in sentenceList)

stop_ids = [dictionary.token2id[stopword] for stopword in stopWords
          if stopword in dictionary.token2id]

dictionary.filter_tokens(stop_ids)
dictionary.compactify()
print(dictionary)

bagOfWords = [dictionary.doc2bow(bigram[line.lower().split()]) for line in sentenceList]

tfidf = models.TfidfModel(bagOfWords)
corpus_tfidf = tfidf[bagOfWords]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
corpus_lsi = lsi[corpus_tfidf]

mycorpus_matrix = matutils.corpus2dense(corpus_lsi, num_terms=30).transpose()

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

numClusters = 3 #calculateNumberOfIdealClusters(40, mycorpus_matrix)

cluster = AgglomerativeClustering(n_clusters=numClusters, linkage="ward", affinity="euclidean")
cluster_labels = cluster.fit_predict(mycorpus_matrix)

print len(cluster_labels)

print len(jobIds)

def calculate_amount_of_clusters_for_each_job_id(cluster_labels, jobIds):

	dictionaryWithClustersForEachJobId = {}

	for index, value in enumerate(cluster_labels):
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

clustersForEachjobSentence = calculate_amount_of_clusters_for_each_job_id(cluster_labels, jobIds)

def writeClustersForEachJobIdToOutputFile(cluster_labels, allJobIds, fileName="output/outputClusterForJobId.csv"):
    with open(fileName, "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["JobId", "Cluster"])
     
        count = 0
        for job_id in allJobIds:
            clusterNumber = cluster_labels[count]
            writer.writerow([job_id, clusterNumber])
            count += 1

def calculateJaccardScoreForEachJobCombo(allJobIds, clustersForEachjobSentence):
	jobCombinationAndCoefficients = []
	index = 0;
	for jobId in list(set(jobIds)):
		copy_list = list(set(jobIds[:])) # only unique jobIds, after that convert back to list
		del copy_list[index] #remove self

		for eachOtherJobId in copy_list:
			otherJobDict = clustersForEachjobSentence[eachOtherJobId]
			currentJobDict = clustersForEachjobSentence[jobId]

			for key in set(cluster_labels):
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

jobCombinationAndCoefficients = calculateJaccardScoreForEachJobCombo(jobIds, clustersForEachjobSentence)

def writeInfoForEachJobIdComboToOutputFile(jobCombinationAndCoefficients, fileName="output/outputInfoForJobIdCombo.csv"):
    with open(fileName, "w+") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["JobId_First", "JobId_Second", "Jaccard_Similarity_Score"])
     
        for job in jobCombinationAndCoefficients:
            writer.writerow([job["job_id_first"], job["job_id_second"], job["jaccard_similarity_score"]])

writeInfoForEachJobIdComboToOutputFile(jobCombinationAndCoefficients)
writeClustersForEachJobIdToOutputFile(cluster_labels, jobIds)


