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
import multiprocessing
from sklearn.linear_model import SGDClassifier
from active_learning import ActiveLearner
from multiprocessing import Process, Queue
print(multiprocessing.cpu_count())

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
    print(Y_train)
    print('leninputx',len(input_X))
    print('lenMu',len(mu))
    print(mu)
    prob_e_step = np.where(np.append(Y_train, mu[Y_train.shape[0]:]) > 0.5, 0, 1)
    #prob_e_step = np.append(Y_train, mu[Y_train.shape[0]:])
   # print(mu[Y_train.shape[0]:])
   # print(prob_e_step)
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

    parser.add_argument("--unlabeled_ideas",
                        type=str,
                        required=True,
                        help="unlabeled_ideas")
    parser.add_argument("--answer_matrix",
                        type=str,
                        required=True,
                        help="answer matrix")
    parser.add_argument("--answer_matrix_viability",
                        type=str,
                        required=True,
                        help="answer matrix viability")
    parser.add_argument("--answer_matrix_feasibility",
                        type=str,
                        required=True,
                        help="answer matrix feasibility")
    parser.add_argument("--answer_matrix_desirability",
                        type=str,
                        required=True,
                        help="answer matrix desirability")

    parser.add_argument("--answer_matrix_unlabeled",
                        type=str,
                        required=True,
                        help="answer matrix _unlabeled")
    parser.add_argument("--answer_matrix_unlabeled_viability",
                        type=str,
                        required=True,
                        help="answer matrix viability _unlabeled")
    parser.add_argument("--answer_matrix_unlabeled_feasibility",
                        type=str,
                        required=True,
                        help="answer matrix feasibility _unlabeled")
    parser.add_argument("--answer_matrix_unlabeled_desirability",
                        type=str,
                        required=True,
                        help="answer matrix desirability _unlabeled")

    parser.add_argument("--sup_rate",
                        default=8,
                        type=int,
                        help="supervision rate")
    parser.add_argument("--max_budget",
                        default=0,
                        type=int,
                        help="max budget")
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

    parser.add_argument("--evaluation_file_feasibility",
                        type=str,
                        required=True,
                        help="feasibility evaluation result after each iteration")
    parser.add_argument("--evaluation_file_desirability",
                        type=str,
                        required=True,
                        help="desirability evaluation result after each iteration")
    parser.add_argument("--evaluation_file_viability",
                        type=str,
                        required=True,
                        help="viability evaluation result after each iteration")
    args = parser.parse_args()
    return args


def var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj, mj, input_X, Y_train, mu,
           classifier_chosen,iterr,idea_dictionnary,evaluation_file):
    vem_step = 0
    theta_i = np.zeros((mu.shape[0], 1))
    while vem_step < iterr:
        mu, sigma_sqr, Aj, Bj, alphaj, mj = e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,idea_dictionnary)

        classifier_chosen, theta_i = m_step(input_X, Y_train, mu, classifier_chosen)

        mu = np.append(Y_train, theta_i[Y_train.shape[0]:])
        # params = m_Step()
        vem_step += 1

    return mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj,classifier_chosen


def main(labeled_ideas,answer_matrix,evaluation_file,iterr,sup_rate,classifier,indices_train,indices_test,answer_matrix_unlabeled,X_unlabeled,max_budget,target_Y):


    n_ideas = labeled_ideas.shape[0]
    n_workers = answer_matrix.worker.unique().size
    sup_rate = 0.1 * sup_rate
    iterr = iterr

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
    # target_Y = labeled_ideas['labels']

    # splitting the data
    # X_train, X_test, Y_train, Y_test = train_test_split(input_X, target_Y,
    #                                                     test_size=(1 - sup_rate), shuffle=False)

    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, shuffle=False)

    # initialize classifier
    choice_classifier = classifier
    if choice_classifier == 'svm':
        classifier_chosen = classifier_init.svm_review_main()
        classifier_chosen.fit(input_X[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(input_X[indices_test])

    elif choice_classifier == 'logreg':
        classifier_chosen = classifier_init.logreg_review_main()
        classifier_chosen.fit(input_X[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(input_X[indices_test])
    elif choice_classifier == 'logreg_emb':
        classifier_chosen, X_train_word_average, X_test_word_average = classifier_init.logreg_embedding(input_X[indices_train], input_X[indices_test])
        classifier_chosen = classifier_chosen.fit(X_train_word_average, target_Y[indices_train])
        Y_pred = classifier_chosen.predict(X_test_word_average)
        input_X = np.concatenate((X_train_word_average, X_test_word_average), axis=0)
    elif choice_classifier == 'bert_based':


        features= classifier_init.bert_based(input_X)
        # X_train, X_test, Y_train, Y_test = train_test_split(features, target_Y,
        #                                                     test_size=(1 - sup_rate), shuffle=False)
        classifier_chosen=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=30, max_iter=5, tol=None)
        classifier_chosen.fit(features[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(features[indices_test])
        input_X=features


    print(classification_report(target_Y[indices_test], Y_pred))
    # print(round(classifier_chosen.score(Y_test, Y_pred) * 100, 2))

    # reporting the results
    report = classification_report(target_Y[indices_test], Y_pred)
    classifier_init.classification_report_csv(report, evaluation_file)
    with open(evaluation_file, 'a') as f:
        f.write('accuracy %s' % accuracy_score(target_Y[indices_test], Y_pred))

    mu = np.append(target_Y[indices_train].values, Y_pred)
    #print(target_Y[indices_train].values)
    #print(Y_pred)
    # print('mu avaliiee',mu)

    mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj,classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj,
                                                        mj, input_X, target_Y[indices_train], mu,
                                                        classifier_chosen, iterr, idea_dictionnary, evaluation_file)

    # print('mu nahayiii', mu)
    # print(round(classifier_chosen.score(target_Y, mu) * 100, 2))
    y_train =target_Y[indices_train]
    predict_theta_i = classifier_chosen.predict(input_X[indices_test])
    print(classification_report(target_Y[indices_test], predict_theta_i))
    with open(evaluation_file, 'a') as f:
        f.write('accuracy %s' % accuracy_score(target_Y[indices_test], predict_theta_i))

    ##active learning

    #answer_matrix_unlabeled = pd.read_csv(args.answer_matrix_unlabeled)
    #Unlabeled_ideas = pd.read_csv(args.inputfile_ideas_unlabeled)
    X_unlabeled = X_unlabeled['idea'].apply(classifier_init.clean_text)
    print("beginning of active learning#########")
    AL = ActiveLearner(strategy='entropy')
    budget=0
    max_budget=5
    X_train = input_X[indices_train]

    while budget < max_budget:
        print('Budgeeeet::::',budget)
        budget=budget+1
        query_index=AL.rank(classifier_chosen, X_unlabeled, num_queries=1)
        print("query_index",query_index)


        Y_pred = classifier_chosen.predict(input_X[indices_test]) # I should change this and give X-text as input

    ##    input_X=np.append(input_X,X_unlabeled[query_index].values)

        print('before',len(X_train))
        X_train=np.append(X_train,X_unlabeled[query_index].values)
        print('afteeerrr',len(X_train))
        print('beforeeinput', len(input_X))
        input_X=np.append(X_train,input_X[indices_test])
        print('afterrinput', len(input_X))
        print('leeeenmuuu',len(mu))
        mu = np.append(mu[:y_train.shape[0]], 0)
        print('leeeenmuuu', len(mu))

        y_train = np.append(y_train, 0)
        # y_train=target_Y[indices_train]




        mu = np.append(mu, Y_pred)
        print('leeeenmuuu', len(mu))



        for indx in query_index:

            # answer_matrix = answer_matrix.append(answer_matrix_unlabeled.iloc[indx] , ignore_index=True)
            temp_answer_matrix=answer_matrix_unlabeled[answer_matrix_unlabeled['idea'] == indx]
            answer_matrix = answer_matrix.append(temp_answer_matrix,ignore_index = True)
            #answer_matrix_unlabeled = answer_matrix_unlabeled[answer_matrix_unlabeled['idea'] == indx]
            #X_unlabeled = X_unlabeled[X_unlabeled.index[indx]]

        #answer_matrix_unlabeled = answer_matrix_unlabeled.drop(answer_matrix_unlabeled[answer_matrix_unlabeled.idea == query_index])

        #print(answer_matrix_unlabeled)
        # answer_matrix_unlabeled_desirability=answer_matrix_unlabeled_desirability.drop(answer_matrix_unlabeled_desirability.index[query_index])
        # answer_matrix_unlabeled_feasibility = answer_matrix_unlabeled_feasibility.drop(answer_matrix_unlabeled_desirability.index[query_index])
        # answer_matrix_unlabeled_desirability = answer_matrix_unlabeled_desirability.drop(answer_matrix_unlabeled_desirability.index[query_index])

        worker_dictionnary = dict(
            zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
        idea_dictionnary = dict(
            zip(answer_matrix['idea'].unique(), np.arange(answer_matrix['idea'].unique().shape[0])))
        #X_unlabeled, y_unlabeled = np.delete(X_unlabeled, query_index, axis=0), np.delete(y_unlabeled, query_index)
        #X_unlabeled= np.delete(X_unlabeled, query_index, axis=0)
        # X_unlabeled = X_unlabeled.drop(X_unlabeled.index[query_index])
        # answer_matrix_unlabeled = answer_matrix_unlabeled[answer_matrix_unlabeled['idea'] == indx]
        #print(X_unlabeled)



        mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj, classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj,
                                                                               sigma_sqr, alphaj,
                                                                               mj, input_X, y_train, mu,
                                                                               classifier_chosen, iterr, idea_dictionnary,
                                                                               target_Y)
        predict_theta_i = classifier_chosen.predict(input_X[indices_test])
        print(classification_report(target_Y[indices_test], predict_theta_i))

        report = classification_report(target_Y[indices_test], predict_theta_i)
        classifier_init.classification_report_csv(report, evaluation_file)
        # with open(evaluation_file, 'a') as f:
        #     f.write('accuracy %s' % accuracy_score(target_Y[indices_test], predict_theta_i))
    return mu,theta_i

    # I think we dont have a y-unlabeled, we just have the answer matrix of the unlabeled samples
    #we can have a X_unlabeled file and delete a row when a criteria runs
    # we can create different answer_matrix, worker_dictionnary for each criteria and run them in multi threads, each create a mu... we can keep all and also at the end mix them
    #making it regression problem: https://www.kaggle.com/sherinclaudia/movie-rating-prediction
    #  target_Y = labeled_ideas['rating']
    #prob_e_step = np.append(Y_train, mu[Y_train.shape[0]:])

#The feasibility of your solution is about whether or not you are able to implement your solution in an effective manner
#The commercial viability of your product or service is an important factor that affects its sustainability and long-term success, will it survive on a longer term?
#Desirability: does it address the user's values and needs
if __name__ == '__main__':
    args_main = parse_args()
    max_budget=args_main.max_budget
    iterr=args_main.iterr
    sup_rate=args_main.sup_rate
    classifier=args_main.classifier
    labeled_ideas = pd.read_csv(
        args_main.inputfile_ideas)  # '/Users/inesarous/Documents/code/peer_idea/input/iclr/small_example/labeled_data.csv'

    X_unlabeled = pd.read_csv(
        args_main.unlabeled_ideas)
    answer_matrix_viability = pd.read_csv(
        args_main.answer_matrix_viability)  # '/Users/inesarous/Documents/code/peer_idea/input/iclr/small_example/answer_matrix.csv'
    answer_matrix_desirability = pd.read_csv(
        args_main.answer_matrix_desirability)
    answer_matrix_feasibility = pd.read_csv(
        args_main.answer_matrix_feasibility)
    answer_matrix_overall = pd.read_csv(
         args_main.answer_matrix)
    # global answer_matrix_unlabeled_feasibility
    # global answer_matrix_unlabeled_desirability
    # global answer_matrix_unlabeled_viability
    # global answer_matrix_unlabeled_overall

    answer_matrix_unlabeled_feasibility= pd.read_csv(
        args_main.answer_matrix_unlabeled_feasibility)
    answer_matrix_unlabeled_desirability = pd.read_csv(
        args_main.answer_matrix_unlabeled_desirability)
    answer_matrix_unlabeled_viability = pd.read_csv(
        args_main.answer_matrix_unlabeled_viability)
    answer_matrix_unlabeled_overall = pd.read_csv(
        args_main.answer_matrix_unlabeled)
    evaluation_file_viability= args_main.evaluation_file_viability  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    #evaluation_file_viability= args.evaluation_file  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    evaluation_file_desirability= args_main.evaluation_file_desirability  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    evaluation_file_feasibility = args_main.evaluation_file_feasibility  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    #evaluation_file_clarity = args.evaluation_file  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'
    evaluation_file_overall = args_main.evaluation_file  # '/Users/inesarous/Documents/code/peer_idea/output/iclr/svm/small_example.csv'

    labels_list = np.char.mod('%d', labeled_ideas['labels'].unique()).tolist()
    # initializing the classification model and cleaning the idea text
    classifier_init = classifier_method()
    # clean the text
    input_X = labeled_ideas['idea'].apply(classifier_init.clean_text)
    # target_Y = labeled_ideas['rating']
    target_Y_viability = labeled_ideas['labels_viability']
    target_Y_feasibility = labeled_ideas['labels_feasibility']
    target_Y_desirability = labeled_ideas['labels_desirability']
    target_Y_overall = labeled_ideas['labels']
    indices=[]
    for ind in range(0, len(target_Y_overall)):
        indices.append(ind)
    print(len(target_Y_overall))
    print(indices)

    # splitting the data
    X_train, X_test, Y_train, Y_test,indices_train,indices_test = train_test_split(input_X, target_Y_overall,indices,
                                                        test_size=0.15, shuffle=False)

    # test_size = (1 - args.sup_rate)
    # Pros=[]
    # Q = Queue()
    mu_feasibility, theta_feasibility=main(labeled_ideas, answer_matrix_feasibility, evaluation_file_feasibility, iterr, sup_rate,
                                   classifier,indices_train,indices_test,answer_matrix_unlabeled_feasibility,X_unlabeled,max_budget,target_Y_feasibility)

    mu_viability, theta_viability=main(labeled_ideas, answer_matrix_viability, evaluation_file_viability, iterr, sup_rate, classifier,indices_train,indices_test,answer_matrix_unlabeled_viability,X_unlabeled,max_budget,target_Y_viability)


    mu_desirability, theta_desirability=main(labeled_ideas, answer_matrix_desirability, evaluation_file_desirability, iterr, sup_rate,
                                   classifier,indices_train,indices_test,answer_matrix_unlabeled_desirability,X_unlabeled,max_budget,target_Y_desirability)


    mu_overall, theta_overall=main(labeled_ideas, answer_matrix_overall, evaluation_file_overall, iterr, sup_rate,
                                   classifier,indices_train,indices_test,answer_matrix_unlabeled_overall,X_unlabeled,max_budget,target_Y_overall)

    print(mu_feasibility, theta_feasibility)
    print(mu_viability, theta_viability)
    print(mu_desirability, theta_desirability)
    print(mu_overall, theta_overall)
    print(np.mean([theta_feasibility, theta_viability, theta_desirability,theta_overall], axis=0))
    # p=Process(target=main,  args=(labeled_ideas, answer_matrix_viability, evaluation_file_viability, iterr, sup_rate, classifier,indices_train,indices_test,answer_matrix_unlabeled_viability,X_unlabeled,max_budget,target_Y_viability))
    # Pros.append(p)
    # p.start()
    # p = Process(target=main, args=(labeled_ideas, answer_matrix_desirability, evaluation_file_desirability, iterr, sup_rate,
    #                                classifier,indices_train,indices_test,answer_matrix_unlabeled_desirability,X_unlabeled,max_budget,target_Y_desirability))
    # Pros.append(p)
    # p.start()
    # p = Process(target=main, args=(labeled_ideas, answer_matrix_feasibility, evaluation_file_feasibility, iterr, sup_rate,
    #                                classifier,indices_train,indices_test,answer_matrix_unlabeled_feasibility,X_unlabeled,max_budget,target_Y_feasibility))
    # Pros.append(p)
    # p.start()
    # # p = Process(target=main), args(labeled_ideas, answer_matrix_clarity, evaluation_file_clarity, iterr, sup_rate,
    # #                                classifier)
    # # Pros.append(p)
    # # p.start()
    # p = Process(target=main, args=(labeled_ideas, answer_matrix_overall, evaluation_file_overall, iterr, sup_rate,
    #                                classifier,indices_train,indices_test,answer_matrix_unlabeled_overall,X_unlabeled,max_budget,target_Y_overall))
    # Pros.append(p)
    # p.start()
    #
    # for t in Pros:
    #     t.join
    # print("QQQQQQQ")
    # print(Q.get())


    # main(labeled_ideas, answer_matrix_viability, evaluation_file, iterr, sup_rate, classifier)
    # main(labeled_ideas, answer_matrix_desirability, evaluation_file, iterr, sup_rate, classifier)
    # main(labeled_ideas, answer_matrix_clarity, evaluation_file, iterr, sup_rate, classifier)
    # main(labeled_ideas, answer_matrix_feasibility, evaluation_file, iterr, sup_rate, classifier)
    # main(labeled_ideas, answer_matrix_feasibility, evaluation_file, iterr, sup_rate, classifier)




    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, shuffle=False)

    # main()
    #make like lorem the end....
