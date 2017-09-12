from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from matplotlib import pyplot as plt
import numpy as np


iris = datasets.load_iris()
print iris.target
# classifier = GaussianNB()
# classifier.fit(iris.data, iris.target)
# score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
# print("Accuracy: %f" % score), iris.target



# petal_len = [round(i[1], 1) for i in iris.data]
# plt.hist(petal_len)
# plt.show()





# get the count of each length
# pe_len_arr = iris.data[:, 0]
# pe_len_arr = [round(i[0], 1) for i in iris.data]

# count_dict = {}
# for len_i in pe_len_arr:
#   if len_i in count_dict:
#     count_dict[len_i] = count_dict[len_i] + 1
#   else:
#     count_dict[len_i] = 1
# print count_dict
