import numpy as np, scipy.sparse, pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

# Loads GloVe embeddings from a designated file location. 
#
# Invoked via:
# ge = GloVe_Embedder(path_to_embeddings)
#
# Embed single word via:
# embed = ge.embed_str(word)
#
# Embed a list of words via:
# embeds = ge.embed_list(word_list)
#
# Find nearest neighbors via:
# ge.find_k_nearest(word, k)
#
# Save vocabulary to file via:
# ge.save_to_file(path_to_file)

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")


# WORKING WITH GloVE and PCA

ge = GloVe_Embedder('GloVe_Embedder_data.txt')
colors = ['blue', 'red', 'green', 'black', 'purple']
word_list = ['flight', 'good', 'terrible', 'help', 'late']
embeds = ge.embed_list(word_list)
word_set = []
labels = []

for i, w in enumerate(word_list):
    word_set.append(w)
    labels.append(i)

    q = ge.find_k_nearest(w, 29)

    for n in q:
        # word_set.add(n[0])
        word_set.append(n[0])
        labels.append(i)

# for w in word_list:
#     word_set.add(w)
print(len(word_set))
print(len(labels))
data = np.ones(len(word_set)) * labels
# data = data.reshape(-1, 1)
print(data.shape)
dataset = np.zeros((len(word_set), 200))
i = 0

for w in word_set:
    dataset[i] = ge.embedding_dict[w]
    i += 1

pca = PCA(2)
new_dataset = pca.fit_transform(dataset)

for i, c in enumerate(labels):
    plt.scatter(new_dataset[i, 0], new_dataset[i, 1], color=colors[c])
plt.ylabel('PCA2')
plt.xlabel('PCA1')

# WORKING WITH GloVE and t-SNE

def get_cluster_indices(w):
    nearest = ge.find_k_nearest(w, 29)
    words = [w[0] for w in nearest]
    c = [i for i in range(len(word_set)) if word_set[i] in words]

    return c

def test_perplexity(ps, data):
    fig, axs = plt.subplots(2, 2, figsize=[16, 12])
    for (i, p) in enumerate(ps):
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=p, random_state=1)
        tsne_data = tsne.fit_transform(data) 
        for k, c in enumerate(cs):
            if i<2:
                axs[0, i-2].scatter(tsne_data[c, 0], tsne_data[c, 1], color=colors[k], label=f"{word_list[k]}")
                axs[0, i-2].set_title(f"perplexity = {p}")
                axs[0, i-2].set_xlabel("principal component 1")
                axs[0, i-2].set_ylabel("principal component 2")
                axs[0, i-2].legend(loc='upper right', prop={'size':8})

            else:
                axs[1, i-2].scatter(tsne_data[c, 0], tsne_data[c, 1], color=colors[k], label=f"{word_list[k]}")
                axs[1, i-2].set_xlabel("principal component 1")
                axs[1, i-2].set_ylabel("principal component 2")
                axs[1, i-2].set_title(f"perplexity = {p}")
                axs[1, i-2].legend(loc='upper right', prop={'size':8})

cs = [get_cluster_indices(word_list[i]) for i in range(5)]
ps = [10, 20, 40, 50]
test_perplexity(ps, dataset)
ps_check = [20, 35, 40, 45]
test_perplexity(ps_check, dataset)

# WORKING WITH K-MEANS CLUSTERING

def purity(labels, gt, k):
    nominator = 0
    NUM_GT_CLUSTERS = 5

    for j in range(k):
        cluster_indices = np.nonzero(labels == j)[0]
        corr = [(gt[cluster_indices] == i).sum() for i in range(NUM_GT_CLUSTERS)]
        nominator += np.max(corr)

    return nominator / gt.shape[0]

def plot_clusters_vs_obj(ks, obj):
    plt.plot(ks,obj)
    plt.ylabel('k-means objective')
    plt.xlabel('number of clusters')
    plt.show()

def plot_clusters_vs_other(ks, other, ylabel):
    plt.plot(ks, other)
    plt.ylabel(ylabel)
    plt.xlabel('number of clusters')
    plt.show()


ks = range(2,21)
objectives = np.zeros((len(ks),1))
labels1 = np.zeros((len(word_set),len(ks)))

for i,k in enumerate(ks):
    q = KMeans(n_clusters=k).fit(dataset)
    objectives[i] = q.inertia_
    labels1[:,i] = q.labels_

gt = np.array(labels)
ars = np.zeros((len(ks),1))
nmi = np.zeros((len(ks),1))
pur = np.zeros((len(ks),1))

for i,k in enumerate(ks):
    nmi[i] = metrics.normalized_mutual_info_score(labels,labels1[:,i])
    ars[i] = metrics.adjusted_rand_score(labels,labels1[:,i])
    pur[i] = purity(labels1[:,i], gt, k)

plot_clusters_vs_obj(ks, objectives)
plot_clusters_vs_other(ks, pur, 'purity score')
plot_clusters_vs_other(ks, ars, 'adjusted random score')
plot_clusters_vs_other(ks, nmi, 'normalized mutual info score')

# IMPROVEMENTS USING WORD EMBEDDINGS
# using SVM on tweet embeddings
def get_embeds(dlist_train):
    tweets_embeds = np.zeros((len(dlist_train),200))
    for k,tweet in enumerate(dlist_train):
        words= dlist_train[k].split()
        wfreq=np.array([words.count(w) for w in words])
        weights = wfreq /wfreq.sum()
        embeds1 = ge.embed_list(words)
        for i in range(embeds1.shape[0]):
            embeds1[i,:]=weights[i]*embeds1[i,:]
        tweets_embeds[k,:] = embeds1.sum(axis=0)/embeds1.shape[0]
    return tweets_embeds

def get_accs_rbf(X_train, y_train, X_val, y_val, C, gamma):
    svm = SVC(C=C, kernel='rbf', gamma=gamma)
    svm.fit(X_train, y_train)
    return svm.score(X_val, y_val), svm.score(X_train, y_train)

def plot_validation_perf(val_accs, gammas, cs):
    sns.heatmap(val_accs, xticklabels=[str(g) for g in gammas], yticklabels=[str(c) for c in cs])
    plt.title('Validation Performance for RBF Kernel')
    plt.xlabel('gamma (log scale)')
    plt.ylabel('C (log scale)')
    plt.show()

def plot_training_perf(train_accs, gammas, cs):
    sns.heatmap(train_accs, xticklabels=[str(g) for g in gammas], yticklabels=[str(c) for c in cs])
    plt.title('Training Performance for RBF Kernel')
    plt.xlabel('gamma (log scale)')
    plt.ylabel('C (log scale)')
    plt.show()

data_train  = pd.read_csv('IA3-train.csv')
data_val    = pd.read_csv('IA3-dev.csv')
dlist_train = data_train['text'].to_list()
y_train     = data_train['sentiment'].to_numpy()
dlist_val   = data_val['text'].to_list()
y_val       = data_val['sentiment'].to_numpy()
X_train     = get_embeds(dlist_train)
X_val       = get_embeds(dlist_val)
cs          = np.arange(-4, 5)
gammas      = np.arange(-5, 2)
val_accs    = np.zeros((cs.shape[0], gammas.shape[0]))
train_accs  = np.zeros((cs.shape[0], gammas.shape[0]))

for i in range(cs.shape[0]):
    c = cs[i]
    
    for j in range(gammas.shape[0]):
        g = gammas[j]
        val_accs[i, j], train_accs[i, j] = get_accs_rbf(X_train, y_train, X_val, y_val, np.power(10.0, c), np.power(10.0, g))

plot_validation_perf(val_accs, gammas, cs)
plot_training_perf(train_accs, gammas, cs)

# averaging weights and running PCA based on sentminent

def add_pca_weights_sentiment(train_data, model):
    data = train_data.copy()
    weight_dims = 100
    data.insert(0, "avg_weight", [np.zeros(weight_dims) for i in range(len(data))])
    
    for i in range(len(data)):
        avg_weights = np.zeros(weight_dims)
        vectorizer = CountVectorizer(lowercase=True)
        bow = vectorizer.fit_transform([data['text'][i]])
        vectors = bow.toarray()
        words = vectorizer.get_feature_names_out()
        for (l, word) in enumerate(words):
            try:
                for (j, w) in enumerate(model[word]):
                    avg_weights[j] += w
            except KeyError:
                np.delete(words, l)

        
        for (k, w) in enumerate(avg_weights):
            avg_weights[k] = avg_weights[k]/len(words)
            
        data['avg_weight'][i]= avg_weights
        
    data.insert(0, "pca", [np.zeros(2) for i in range(len(data))])
    pca = PCA(2)
    pca_data = np.zeros((len(data), weight_dims))
    for i in range(len(data)):
        pca_data[i] = data['avg_weight'][i]
    new_dataset = pca.fit_transform(pca_data)
    
    for i in range(len(data)):
        data['pca'][i] = new_dataset[i]
        
    return data

def get_sentiment_clusters(data):
    positive = []
    negative = []
    for i in range(len(data)):
        sentiment = data['sentiment'][i]
        if sentiment == 1:
            positive.append(data['pca'][i])
        else:
            negative.append(data['pca'][i])
    return (positive, negative)

def plot_sentiment_pca(new_data):
    fig, ax = plt.subplots(figsize=[16,9])
    colors = ['blue', 'red']
    sentiment = ['positive', 'negative']
    sentiment_data_set = get_sentiment_clusters(new_data)
    for (i, pca_data) in enumerate(sentiment_data_set):        
        plt.scatter([pca_data[j][0] for j in range(len(pca_data))],[pca_data[j][1] for j in range(len(pca_data))], color=colors[i], label=f"{sentiment[i]}")
        
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    ax.legend(loc='upper right')
    plt.show()

model_gigaword = api.load("glove-wiki-gigaword-100")
new_sentiment_data = add_pca_weights_sentiment(data_train, model_gigaword)
plot_sentiment_pca(new_sentiment_data)
            