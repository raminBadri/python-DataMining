import pandas as pd
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
import random

# initialize
path = "C:/Users/vaio/Desktop/ramin-project/source/TripAdvisor_First_100_Hotels"
token_dict = {}
stopwords = nltk.corpus.stopwords.words('english')
stemmer = PorterStemmer()
document_names = []


def tokenize_and_stem_and_removestopwords(input_text):  # Tokenize, Stem and remove stopwords
    tokens = [word for sent in nltk.sent_tokenize(input_text) for word in nltk.word_tokenize(sent)]
    filtered_stopwords = [w for w in tokens if w not in stopwords]
    filtered_tokens = []
    for token in filtered_stopwords:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# K-means
def k_means(tfidfvector, numofclusters):
    cluster_id = []
    doc_id_per_cluster = []
    km = KMeans(n_clusters=numofclusters)
    km.fit(tfidfvector)
    clusters = km.labels_.tolist()
    for k in range(0, numofclusters):
        cluster_id.append([])
        doc_id_per_cluster.append([])
    for k in range(0, len(clusters)):
        for i in range(0, numofclusters):
            if clusters[k] == i:
                cluster_id[i].append(clusters[k])
                doc_id_per_cluster[i].append(k)
    return clusters, cluster_id, doc_id_per_cluster


def show_result(result_array, clustercounts):
    print("**There are ", len(result_array[0]), " documents in TripAdvisor data collection**")
    print("**The data collection is clustered in ", clustercounts, "clusters as shown in below**")
    print("##################")
    for j in range(0, len(result_array[1])):
        print("There are: ", len(result_array[1][j]), " documents in cluster: ", j)
        print("Here we can see names of the documents:")
        print()
        for m in result_array[2][j]:
            print(document_names[m])
        print()
        print("==================")


def draw_shape(a, b, c, d):
    dist = 1 - cosine_similarity(c)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]
    cluster_colors = []
    cluster_names = []
    prefix = "cluster_"
    for i in range(0, d):
        cluster_colors.append(colorize())
        cluster_names.append(prefix+str(i))
    df = pd.DataFrame(dict(x=xs, y=ys, label=a, title=b))
    groups = df.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')
    ax.legend(numpoints=1)  # show legend with only 1 point
    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=5)
    plt.show()
    plt.close()

    # dendrogram plot
    linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="top", labels=document_names, leaf_font_size=6)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom='on')
    plt.show()
    plt.close()
    return True


def colorize():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())

# iterate throw data collection
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, encoding="utf8")
        new_path = file_path.replace(file_path, file_path[72:90])
        document_names.append(new_path)
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(string.punctuation)
        token_dict[file] = no_punctuation

flag = True
ans_1 = input("Please enter number of clusters(k for k-means): ")
while flag:
    try:
        val = int(ans_1)
        flag = False
    except ValueError:
        print("That's not an int!")
tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem_and_removestopwords, stop_words='english', ngram_range=(1, 2))
tfs = tfidf.fit_transform(token_dict.values())
result = k_means(tfs, int(ans_1))
show_result(result, int(ans_1))
ans_2 = input("Do you want to see clustering visual shapes? y or n: ")
if ans_2 == "y":
    draw_shape(result[0], document_names, tfs, int(ans_1))
# THE END