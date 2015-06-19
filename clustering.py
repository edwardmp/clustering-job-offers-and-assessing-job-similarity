import logging
import numpy as np
import csv
from gensim import corpora, models, similarities, matutils
from sklearn.cluster import AgglomerativeClustering
from math import log
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loadStopWords():
    stopWords = [line.rstrip('\n\r') for line in open('stop-words_dutch.txt')]
    stopWords = set(stopWords) # only unique elements
    print "Number of Dutch Stop Words loaded: %d\n" % len(stopWords)
    return stopWords

def openInputDataFileAndReturnSentencesRows(fileName='input/data.csv'):
    file = open(fileName)
    csvFileReader = csv.reader(file)

    # skip column header
    next(csvFileReader, None)

    jobs = []
    jobIds = []
    index = 0
    for line in csvFileReader:
        jobIds.append(line[0])
        jobs.append(line[1])

    return (jobs, jobIds)

def openInputDataFileContainingJobTitlesForJobIdsAndReturnDictionary(fileName='input/jobTitlesForJobIds.csv'):
    file = open(fileName)
    csvFileReader = csv.reader(file)

    # skip column header
    next(csvFileReader, None)

    dictionaryContainingJobTitleForJobIdKey = {}
    for line in csvFileReader:
		dictionaryContainingJobTitleForJobIdKey[line[0]] = line[1]

    return dictionaryContainingJobTitleForJobIdKey

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
        writer.writerow(["Job_Id_First", "Job_Id_Second", "Jaccard_Similarity_First_Method_Score", "Jaccard_Similarity_Second_Method_Score", "Jaccard_Similarity_Difference", "Job_Title_First", "Job_Title_Second"])

        for job in jobCombinationAndCoefficients:
            writer.writerow([job["job_id_first"], job["job_id_second"], job["jaccard_similarity_first_method_score"], job["jaccard_similarity_second_method_score"], job["jaccard_similarity_difference"], job["job_title_first"], job["job_title_second"]])

def calculateNumberOfIdealClusters(maxAmount, corpus):
	print "Initializing silhouette analysis"
	range_n_clusters = range(2, maxAmount) # max amount of clusters equal to amount of jobs

	silhouette_high = 0;
	silhouette_high_n_clusters = 2;

	for n_clusters in range_n_clusters:
		# Initialize the clusterer with n_clusters value
		cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity="euclidean")
		cluster_labels = cluster.fit_predict(corpus)

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(corpus, cluster_labels)

		print "For n_clusters = %d, the average silhouette_score is: %.5f" % (n_clusters, silhouette_avg)

		if (silhouette_avg > silhouette_high):
		    silhouette_high = silhouette_avg
		    silhouette_high_n_clusters = n_clusters

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(corpus, cluster_labels)

	print ("Highest score = %f for n_clusters = %d" % (silhouette_high, silhouette_high_n_clusters))
	return silhouette_high_n_clusters

def calculateAmountOfClustersForEachJobId(clusterLabels, jobIds):
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

def calculateJaccardFirstMethod(firstJobSentencesPerCluster, secondJobSentencesPerCluster):
    countIntersectionNumberOfClusters = 0
    for index, elt in enumerate(firstJobSentencesPerCluster):
        if secondJobSentencesPerCluster[index] != 0 and elt != 0:
            countIntersectionNumberOfClusters += 1

    amountOfUniqueClustersPresent = max(len(filter(lambda a: a != 0, firstJobSentencesPerCluster)), len(filter(lambda a: a != 0, secondJobSentencesPerCluster)))

    return (float(countIntersectionNumberOfClusters) / float(amountOfUniqueClustersPresent))

def calculateJaccardSecondMethod(currentJobDict, otherJobDict):
    unionCount = 0
    intersectionCount = 0

    for key in currentJobDict.keys():
        intersectionCount = intersectionCount + min(currentJobDict[key], otherJobDict[key])
        unionCount = unionCount + max(currentJobDict[key], otherJobDict[key])

    coefficient = float(intersectionCount) / float(unionCount)

    return coefficient

def calculateJaccardScoresForEachJobCombo(allJobIds, clustersForEachjobSentence, clusterLabels, jobTitlesForJobIds):
    jobCombinationAndCoefficients = []
    index = 0;
    for jobId in list(set(jobIds)):
        copy_list = list(set(jobIds[:])) # only unique jobIds, after that convert back to list
        del copy_list[index] #remove self

        for eachOtherJobId in copy_list:
            otherJobDict = clustersForEachjobSentence[eachOtherJobId]
            currentJobDict = clustersForEachjobSentence[jobId]

            # add clusters that do not have sentences for that job id in them
            for key in set(clusterLabels):
                if key not in currentJobDict:
                    currentJobDict[key] = 0

                if key not in otherJobDict:
                    otherJobDict[key] = 0

            coefficientFirstJaccardLikeMethod = round(calculateJaccardFirstMethod(currentJobDict.values(), otherJobDict.values()), 3)
            coefficientSecondJaccardLikeMethod = round(calculateJaccardSecondMethod(currentJobDict, otherJobDict), 3)
            coefficientDifference = round((max(coefficientSecondJaccardLikeMethod, coefficientFirstJaccardLikeMethod) - min(coefficientFirstJaccardLikeMethod, coefficientSecondJaccardLikeMethod)), 3)
            jobTitleFirst = jobTitlesForJobIds[jobId]
            jobTitleSecond = jobTitlesForJobIds[eachOtherJobId]

            # create output file containg all below data, for reference
            jobCombinationAndCoefficients.append({"job_id_first": jobId, "job_id_second": eachOtherJobId, "jaccard_similarity_first_method_score": coefficientFirstJaccardLikeMethod, "jaccard_similarity_second_method_score": coefficientSecondJaccardLikeMethod, "jaccard_similarity_difference": coefficientDifference, "job_title_first": jobTitleFirst, "job_title_second": jobTitleSecond})

            print "First job with title %s (key = cluster identier, value = amount of sentences in cluster): %s" % (jobTitleFirst, currentJobDict)
            print "Second job with title %s (key = cluster identier, value = amount of sentences in cluster): %s" % (jobTitleSecond, otherJobDict)
            print "Job with ids %s and %s are (first method) %.3f similar and (second method) %.3f similar" % (jobId, eachOtherJobId, coefficientFirstJaccardLikeMethod, coefficientSecondJaccardLikeMethod)

            if (coefficientDifference >= float(0.500)):
                print "Difference between both Jaccard scores is %.3f, more than 0.500 threshold" %  coefficientDifference

            # Newline for clarity reasons. You're welcome.
            print ""

        index += 1

    return jobCombinationAndCoefficients

# some unit tests to confirm both Jaccard scores are calculated correctly
def testIfFirstJaccardLikeSimilarityCalculationIsDoneCorrectly():
    # first vacancy has 7 sentences in 3 different clusters:
    # 2 sentences in cluster 0, 4 sentences in cluster 1 and 1 sentence in cluster 2
    firstVacancyNumberOfSentencesInEachClusterDictionary = {0: 2, 1: 4, 2: 1}

    # second vacancy has 7 sentences in 3 different clusters:
    # 1 sentence in cluster 0, 3 sentences in cluster 1 and 2 sentences in cluster 2
    secondVacancyNumberOfSentencesInEachClusterDictionary = {0: 1, 1: 3, 2: 2}

    # from manual calculation we know the Jaccard like score for this should be 5 divided by 8
    coefficient = calculateJaccardFirstMethod(firstVacancyNumberOfSentencesInEachClusterDictionary.values(), secondVacancyNumberOfSentencesInEachClusterDictionary.values())

    if coefficient != (float(3)/float(3)):
        exceptionContent = 'Test that confirms calculation of first Jaccard type coefficient is done right has failed, result is not equal to 1 (which is %f) but is %f' % ((float(3)/float(3)), coefficient)
        raise AssertionError(exceptionContent)
    else:
        print 'Successfully confirmed that first Jaccard like similarity is calculated correctly!'

def testIfSecondJaccardLikeSimilarityCalculationIsDoneCorrectly():
    # first vacancy has 7 sentences in 3 different clusters:
    # 2 sentences in cluster 0, 4 sentences in cluster 1 and 1 sentence in cluster 2
    firstVacancyNumberOfSentencesInEachClusterDictionary = {0: 2, 1: 4, 2: 1}

    # second vacancy has 7 sentences in 3 different clusters:
    # 1 sentence in cluster 0, 3 sentences in cluster 1 and 2 sentences in cluster 2
    secondVacancyNumberOfSentencesInEachClusterDictionary = {0: 1, 1: 3, 2: 2}

    # from manual calculation we know the Jaccard like score for this should be 5 divided by 8
    coefficient = calculateJaccardSecondMethod(firstVacancyNumberOfSentencesInEachClusterDictionary, secondVacancyNumberOfSentencesInEachClusterDictionary)

    if coefficient != (float(5)/float(8)):
        exceptionContent = 'Test that confirms calculation of second Jaccard type coefficient is done right has failed, result is not equal to 5/8 (which is %f) but is %f' % ((float(5)/float(8)), coefficient)
        raise AssertionError(exceptionContent)
    else:
        print 'Successfully confirmed that second Jaccard like similarity is calculated correctly!'

testIfFirstJaccardLikeSimilarityCalculationIsDoneCorrectly()
testIfSecondJaccardLikeSimilarityCalculationIsDoneCorrectly()

sentenceList, jobIds = openInputDataFileAndReturnSentencesRows()

jobTitlesForJobIds = openInputDataFileContainingJobTitlesForJobIdsAndReturnDictionary()

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

clustersForEachjobSentence = calculateAmountOfClustersForEachJobId(clusterLabels, jobIds)

jobCombinationAndCoefficients = calculateJaccardScoresForEachJobCombo(jobIds, clustersForEachjobSentence, clusterLabels, jobTitlesForJobIds)

writeInfoForEachJobIdComboToOutputFile(jobCombinationAndCoefficients)
writeClustersForEachJobIdToOutputFile(clusterLabels, jobIds, sentenceList)
