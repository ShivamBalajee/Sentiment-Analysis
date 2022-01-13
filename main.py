# STOP_WORDS
# stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

import math
import random
from matplotlib import pyplot as plt


def process_text(data):
    data = data.strip().lower()
    punct = '''!()-[]{};:'"\,<>./?@#$%^&*...!!!..!!_~'''
    for item in punct:
        data = data.replace(item, '')
    return data.split()


def load_data(path_to_file):
    List = []
    with open(path_to_file) as f:
        contents = f.readlines()
        for item in contents:
            a = item.split('\t')

            List.append((process_text(a[0]), a[1].replace('\n', '')))

        random.shuffle(List)
        return List


def k_fold(data, k):
    n = len(data)
    ans = []
    chunk_size = int(n / k)
    for i in range(k):
        ans.append((data[:i * chunk_size] + data[(i + 1) * chunk_size:], data[i * chunk_size:(i+1)*chunk_size]))
    return ans


def sentence_label(List):
    X = []
    y = []
    for sen, lab in List:
        X.append(sen)
        y.append(lab)
    return X, y


class Naive_Bayes:
    def __init__(self, m=0):
        self.m = m

    def train(self, X, y):
        count_zero = 0
        count_one = 0
        total_count = len(X)
        for lab in y:
            if lab == '0':
                count_zero += 1
            elif lab == '1':
                count_one += 1
        dict_zeros = {}
        dict_ones = {}

        for sentence, label in zip(X, y):
            for word in sentence:
                if label == '1':
                    if word in dict_ones:
                        dict_ones[word] += 1
                    else:
                        dict_ones[word] = 1

                elif label == '0':
                    if word in dict_zeros:
                        dict_zeros[word] += 1
                    else:
                        dict_zeros[word] = 1

        self.probablity_class_zero = count_zero / total_count
        self.probablity_class_one = count_one / total_count
        self.dictionary_one = dict_ones
        self.dictionary_zero = dict_zeros

    def test(self, X):
        total_one = sum(self.dictionary_one.values())
        total_zero = sum(self.dictionary_zero.values())
        y = []
        v = len(self.dictionary_zero) + len(self.dictionary_one)
        for sentence in X:
            probablity_num_zero = 0
            probablity_num_one = 0
            for word in sentence:
                try:
                    probablity_num_zero += math.log(
                        (self.dictionary_zero.get(word, 0) + self.m) / (total_zero + self.m*v))
                except(ValueError):
                    probablity_num_zero = -math.inf
                try:
                    probablity_num_one += math.log(
                        (self.dictionary_one.get(word, 0) + self.m) / (total_one + self.m*v))
                except(ValueError):
                    probablity_num_one = -math.inf

            probablity_0 = math.log(self.probablity_class_zero) + probablity_num_zero
            probablity_1 = math.log(self.probablity_class_one) + probablity_num_one

            if probablity_1 >= probablity_0:
                y.append('1')
            elif probablity_0 > probablity_1:
                y.append('0')
        return y


def accuracy(pred, test_y):
    count = 0
    for i, j in zip(pred, test_y):
        if i == j:
            count += 1
    return count / len(test_y)


# model.train(list(map(lambda x: x.split(),['what a nice day','a green cat chased a green dog', 'green umbrella a nice day'])),['1','1','0'])
# print(model.test(map(lambda x: x.split(),['a nice day','what a green umbrella'])))

arr = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
for i in arr:
    List_data = load_data('pp1data/' + i)

    for train, test in k_fold(List_data, 10):
        train_x, train_y = sentence_label(train)
        test_x, test_y = sentence_label(test)
        model = Naive_Bayes(m=10)
        model.train(train_x, train_y)
        pred = model.test(test_x)
        print(accuracy(pred, test_y))



for i in arr:
    List_data = load_data('pp1data/' + i)

    results = {
        0.1: [],
        0.2:[],
        0.3:[],
        0.4:[],
        0.5:[],
        0.6:[],
        0.7:[],
        0.8:[],
        0.9:[],
        1:[]
    }
    for train, test in k_fold(List_data, 10):
        train_x, train_y = sentence_label(train)
        test_x, test_y = sentence_label(test)
        N = len(train_x)

        for i in range(1,11):
            size = i/10
            small_train_x = train_x[:int(size*N)]
            small_train_y = train_y[:int(size*N)]
            model = Naive_Bayes(m=1)
            model.train(small_train_x, small_train_y)
            preds = model.test(test_x)
            acc = accuracy(preds, test_y)
            results[round(i/10,1)].append(acc)
    x = []
    y = []
    for i, j in results.items():
        x.append(i)
        avg = sum(j)/len(j)
        y.append(avg)

    plt.plot(x, y)
    plt.show()

for file in arr:
    List_data = load_data('pp1data/' + file)

    results_m0 = {
        0.1: [],
        0.2:[],
        0.3:[],
        0.4:[],
        0.5:[],
        0.6:[],
        0.7:[],
        0.8:[],
        0.9:[],
        1:[]
    }
    results_m1 = {
        0.1: [],
        0.2: [],
        0.3: [],
        0.4: [],
        0.5: [],
        0.6: [],
        0.7: [],
        0.8: [],
        0.9: [],
        1: []
    }
    for train, test in k_fold(List_data, 10):
        train_x, train_y = sentence_label(train)
        test_x, test_y = sentence_label(test)
        N = len(train_x)

        for i in range(1,11):
            size = i/10
            small_train_x = train_x[:int(size*N)]
            small_train_y = train_y[:int(size*N)]
            model = Naive_Bayes(m=0)
            model.train(small_train_x, small_train_y)
            preds = model.test(test_x)
            acc = accuracy(preds, test_y)
            results_m0[round(i/10,1)].append(acc)

            model = Naive_Bayes(m=1)
            model.train(small_train_x, small_train_y)
            preds = model.test(test_x)
            acc = accuracy(preds, test_y)
            results_m1[round(i / 10, 1)].append(acc)
    x_m0 = []
    y_m0 = []
    for i, j in results_m0.items():
        x_m0.append(i)
        avg = sum(j)/len(j)
        y_m0.append(avg)

    plt.plot(x_m0, y_m0, label='m=0')

    x_m1 = []
    y_m1 = []
    for i, j in results_m1.items():
        x_m1.append(i)
        avg = sum(j) / len(j)
        y_m1.append(avg)

    plt.plot(x_m1, y_m1, label='m=1')
    plt.legend()
    plt.suptitle(file)
    plt.show()


m_s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,2,3,4,5,6,7,8,9,10]

for i in arr:
    List_data = load_data('pp1data/' + i)

    results = {    }
    for m in m_s:
        accs = []
        for train, test in k_fold(List_data, 10):
            train_x, train_y = sentence_label(train)
            test_x, test_y = sentence_label(test)

            model = Naive_Bayes(m=m)
            model.train(train_x, train_y)
            preds = model.test(test_x)
            acc = accuracy(preds, test_y)
            accs.append(acc)
            print(accs)
        results[m]=sum(accs)/len(accs)
    print(results)
    x = []
    y = []
    for i, j in results.items():
        x.append(str(i))
        y.append(j)

    plt.bar(x, y)
    plt.show()