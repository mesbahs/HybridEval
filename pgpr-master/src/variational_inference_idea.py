import pandas as pd
import numpy as np
from classifier import classifier_method
from IPython import embed
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from active_learning import ActiveLearner



def init_probabilities(n_reviews):
    # initialize probability z_i (item's quality) randomly
    sigma_sqr = np.ones((n_reviews, 1))
    # initialize probability alpha beta (worker's reliability)
    A = 2
    B = 2
    alpha = 1
    return sigma_sqr, A, B, alpha


def init_Aj_Bj(A, B, n_workers,alpha):
    Aj = A * np.ones((n_workers, 1), dtype='float32')
    Bj = B * np.ones((n_workers, 1), dtype='float32')
    alphaj = alpha * np.ones((n_workers, 1), dtype='float32')
    mj = np.zeros((n_workers, 1), dtype='float32')
    return Aj, Bj, alphaj, mj


def e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,idea_dictionnary):
    # start E step
    # updating z_i
    all_ideas_id = answer_matrix['idea'].unique()
    all_workers_id = answer_matrix['worker'].unique()
    for i in all_ideas_id:
        W = 0.0
        V = 0.0
        i_index = idea_dictionnary[i]
        answers_i = answer_matrix[answer_matrix['idea'] == i]
        workers_i = answers_i['worker'].unique()
        for j in workers_i:

            j_index = worker_dictionnary[j]
            W = W + (Aj[j_index, 0] / Bj[j_index, 0]) * (answers_i[answers_i['worker'] == j].iloc[0, 2] - mj[j_index])
            V = V + (Aj[j_index, 0] / Bj[j_index, 0])
        W = W + (mu[i_index] / sigma_sqr[i_index, 0])
        V = V + (1 / sigma_sqr[i_index])
        mu[i_index] = W / V
        sigma_sqr[i_index] = 1 / V
    for j in all_workers_id:
        j_index = worker_dictionnary[j]
        answers_j = answer_matrix[answer_matrix['worker'] == j]
        ideas_j = answers_j['idea'].unique()
        X = Aj[j_index] + 0.5
        Y = Bj[j_index] + 0.5 * (ideas_j.shape[0] / alphaj[j_index])
        for i in ideas_j:
            i_index = idea_dictionnary[i]
            Y = Y + 0.5 * ((answers_j[answers_j['idea'] == i].iloc[0, 2] ** 2) + sigma_sqr[i_index, 0] - (
                    2 * answers_j[answers_j['idea'] == i].iloc[0, 2] * mu[i_index]) - (
                                   2 * answers_j[answers_j['idea'] == i].iloc[0, 2] * mj[j_index]) + (
                                   2 * mu[i_index] * mj[j_index]))
        Aj[j_index] = X
        Bj[j_index] = Y
        K = (ideas_j.shape[0] * (Aj[j_index] / Bj[j_index])) + alphaj[j_index]
        L = 0
        for i in ideas_j:
            i_index = idea_dictionnary[i]
            L = L + (answers_j[answers_j['idea'] == i].iloc[0, 2] - mu[i_index])
        L = (Aj[j_index] / Bj[j_index]) * L
        alphaj[j_index] = 1 / K
        mj[j_index] = L / K
    return mu, sigma_sqr, Aj, Bj, alphaj, mj


def m_step(input_X, Y_train, mu, classifier_chosen):
    # start M step
    print('lenY_train', len(Y_train))
    print('leninputx',len(input_X))
    print('lenMu',len(mu))
    prob_e_step = np.where(np.append(Y_train, mu[Y_train.shape[0]:]) > 0.5, 0, 1)
    #prob_e_step = np.append(Y_train, mu[Y_train.shape[0]:])
    print(mu[Y_train.shape[0]:])
    print(prob_e_step)
    classifier_chosen = classifier_chosen.fit(input_X, prob_e_step)
    theta_i = classifier_chosen.predict(input_X)

    return classifier_chosen, theta_i


def parse_args():
    parser = argparse.ArgumentParser(
        description="naive bayes")
    parser.add_argument("--inputfile_ideas",
                        type=str,
                        required=True,
                        help="inputfile ideas")
    parser.add_argument("--answer_matrix",
                        type=str,
                        required=True,
                        help="answer matrix")
    parser.add_argument("--sup_rate",
                        default=8,
                        type=int,
                        help="supervision rate")
    parser.add_argument("--iterr",
                        default=10,
                        type=int,
                        help="number of EM iterations")

    parser.add_argument("--classifier",
                        type=str,
                        choices=['svm', 'logreg', 'logreg_emb','bert_based'],
                        required=True,
                        help="choice of classifier")

    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    args = parser.parse_args()
    return args


def var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj, mj, input_X, Y_train, mu,
           classifier_chosen,iterr,idea_dictionnary,target_Y):
    vem_step = 0
    theta_i = np.zeros((mu.shape[0], 1))
    while vem_step < iterr:
        mu, sigma_sqr, Aj, Bj, alphaj, mj = e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,idea_dictionnary)
        print(mu)
        classifier_chosen, theta_i = m_step(input_X, Y_train, mu, classifier_chosen)
        # print(classification_report(Y_test, theta_i))
        print(theta_i)

        mu = np.append(Y_train.values, theta_i[Y_train.shape[0]:])
        # params = m_Step()
        vem_step += 1

    return mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj,classifier_chosen


def main():
    args = parse_args()
    labeled_ideas = pd.read_csv(
        args.inputfile_ideas)  # '/Users/inesarous/Documents/code/peer_idea/input/iclr/small_example/labeled_data.csv'
    answer_matrix = pd.read_csv(
        args.answer_matrix)  # '/Users/inesarous/Documents/code/peer_idea/input/iclr/small_example/answer_matrix.csv'
    answer_matrix['rating'] = (answer_matrix['rating'] - 1) / 4
    evaluation_file = args.evaluation_file  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    labels_list = np.char.mod('%d', labeled_ideas['labels'].unique()).tolist()

    n_ideas = labeled_ideas.shape[0]
    n_workers = answer_matrix.worker.unique().size
    sup_rate = 0.1 * args.sup_rate
    iterr = args.iterr

    # initializing the parameters
    sigma_sqr, A, B, alpha = init_probabilities(n_ideas)
    Aj, Bj, alphaj, mj = init_Aj_Bj(A, B, n_workers,alpha)

    worker_dictionnary = dict(
        zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
    idea_dictionnary = dict(
        zip(answer_matrix['idea'].unique(), np.arange(answer_matrix['idea'].unique().shape[0])))

    # initializing the classification model and cleaning the idea text
    classifier_init = classifier_method()
    # clean the text
    input_X = labeled_ideas['idea'].apply(classifier_init.clean_text)
    #target_Y = labeled_ideas['rating']
    target_Y = labeled_ideas['labels']

    # splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(input_X, target_Y,
                                                        test_size=(1 - sup_rate), shuffle=False)

    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, shuffle=False)

    # initialize classifier
    choice_classifier = args.classifier
    if choice_classifier == 'svm':
        classifier_chosen = classifier_init.svm_review_main()
        classifier_chosen.fit(X_train, Y_train)
        Y_pred = classifier_chosen.predict(X_test)
    elif choice_classifier == 'logreg':
        classifier_chosen = classifier_init.logreg_review_main()
        classifier_chosen.fit(X_train, Y_train)
        Y_pred = classifier_chosen.predict(X_test)
    elif choice_classifier == 'logreg_emb':
        classifier_chosen, X_train_word_average, X_test_word_average = classifier_init.logreg_embedding(X_train, X_test)
        classifier_chosen = classifier_chosen.fit(X_train_word_average, Y_train)
        Y_pred = classifier_chosen.predict(X_test_word_average)
        input_X = np.concatenate((X_train_word_average, X_test_word_average), axis=0)
    elif choice_classifier == 'bert_based':


        features= classifier_init.bert_based(input_X)
        X_train, X_test, Y_train, Y_test = train_test_split(features, target_Y,
                                                            test_size=(1 - sup_rate), shuffle=False)
        classifier_chosen=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=30, max_iter=5, tol=None)
        classifier_chosen.fit(X_train, Y_train)
        Y_pred = classifier_chosen.predict(X_test)
        input_X=features


    print(classification_report(Y_test, Y_pred))
    # print(round(classifier_chosen.score(Y_test, Y_pred) * 100, 2))

    # reporting the results
    report = classification_report(Y_test, Y_pred)
    classifier_init.classification_report_csv(report, evaluation_file)
    with open(evaluation_file, 'a') as f:
        f.write('accuracy %s' % accuracy_score(Y_test, Y_pred))

    mu = np.append(Y_train.values, Y_pred)
    print(Y_train.values)
    print(Y_pred)
    print('mu avaliiee',mu)

    mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj,classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj,
                                                        mj, input_X, Y_train, mu,
                                                        classifier_chosen, iterr, idea_dictionnary, target_Y)

    print('mu nahayiii', mu)
    # print(round(classifier_chosen.score(target_Y, mu) * 100, 2))
    print(classification_report(target_Y, mu))

    ##active learning

    #answer_matrix_unlabeled = pd.read_csv(args.answer_matrix_unlabeled)
    #Unlabeled_ideas = pd.read_csv(args.inputfile_ideas_unlabeled)
    #X_unlabeled = Unlabeled_ideas['idea'].apply(classifier_init.clean_text)

    #AL = ActiveLearner(strategy='entropy')
    #budget=0
    #max_budget=500
    #while budget < max_budget:
        # query_index=AL.rank(classifier_chosen, X_unlabeled, num_queries=1)
        #
        # Y_pred = classifier_chosen.predict(X_unlabeled[query_index])
        # input_X=np.append(input_X,X_unlabeled[query_index])
        # mu = np.append(mu, Y_pred)


        #for indx in query_index:

            #answer_matrix = answer_matrix.append(answer_matrix_unlabeled.iloc[indx] , ignore_index=True)

        #answer_matrix_unlabeled = answer_matrix_unlabeled.drop(df.index[query_index])

        # worker_dictionnary = dict(
        #     zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
        # idea_dictionnary = dict(
        #     zip(answer_matrix['idea'].unique(), np.arange(answer_matrix['idea'].unique().shape[0])))
        # X_unlabeled, y_unlabeled = np.delete(X_unlabeled, query_index, axis=0), np.delete(y_unlabeled, query_index)



        # mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj, classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj,
        #                                                                        sigma_sqr, alphaj,
        #                                                                        mj, input_X, Y_train, mu,
        #                                                                        classifier_chosen, iterr, idea_dictionnary,
        #                                                                        target_Y)

    # I think we dont have a y-unlabeled, we just have the answer matrix of the unlabeled samples
    #we can have a X_unlabeled file and delete a row when a criteria runs
    # we can create different answer_matrix, worker_dictionnary for each criteria and run them in multi threads, each create a mu... we can keep all and also at the end mix them
    #making it regression problem: https://www.kaggle.com/sherinclaudia/movie-rating-prediction
    #  target_Y = labeled_ideas['rating']
    #prob_e_step = np.append(Y_train, mu[Y_train.shape[0]:])


if __name__ == '__main__':
    main()
