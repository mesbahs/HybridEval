import pandas as pd
import numpy as np
from classifier import classifier_method
from IPython import embed
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix,recall_score, precision_score,f1_score
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import statistics
import torch
import transformers as ppb

import pandas as pd
import multiprocessing
from sklearn.linear_model import SGDClassifier
from active_learning import ActiveLearner
from multiprocessing import Process, Queue
from sklearn.utils.validation import check_X_y
import random
print(multiprocessing.cpu_count())
from sklearn.metrics import mean_squared_error

import math
def init_probabilities(n_reviews):
    # initialize probability z_i (item's quality) randomly
    sigma_sqr = np.ones((n_reviews, 1))
    # initialize probability alpha beta (worker's reliability)
    A = 2
    B = 2
    alpha = 1
    return sigma_sqr, A, B, alpha
def RBO(l1, l2, p=0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


def init_Aj_Bj(A, B, n_workers,alpha):
    Aj = A * np.ones((n_workers, 1), dtype='float32')
    Bj = B * np.ones((n_workers, 1), dtype='float32')
    alphaj = alpha * np.ones((n_workers, 1), dtype='float32')
    mj = np.zeros((n_workers, 1), dtype='float32')
    return Aj, Bj, alphaj, mj
def measure_rbo(classifier_desirability, classifier_feasibility, classifier_viability, classifier_overall,labeled_ideas,indices_test,classifier):
    classifier_init = classifier_method()
    labeled_ideas=labeled_ideas.iloc[indices_test]

    input_X = labeled_ideas['idea'].apply(classifier_init.clean_text)
    input_id = labeled_ideas['Id']
    if classifier == 'bert_based':
        input_X = prepare_bert(input_X)

    print('classifier_desirability.predict(input_X):::::::')
    print(classifier_desirability.predict(input_X))

    labeled_ideas['desirability'] = classifier_desirability.predict(input_X)
    print(labeled_ideas['desirability'].values)
    labeled_ideas['feasibility'] = classifier_feasibility.predict(input_X )
    labeled_ideas['viability'] = classifier_viability.predict(input_X )
    labeled_ideas['overall'] = classifier_overall.predict(input_X )
    labeled_ideas[
        'multiply_all'] = labeled_ideas.desirability * labeled_ideas.feasibility * labeled_ideas.viability * labeled_ideas.overall
    labeled_ideas[
        'sum_all'] = labeled_ideas.desirability + labeled_ideas.feasibility + labeled_ideas.viability +labeled_ideas.overall
    labeled_ideas['sum_3'] = labeled_ideas.desirability + labeled_ideas.feasibility + labeled_ideas.viability
    labeled_ideas['ground_truth'] = labeled_ideas.labels*labeled_ideas.labels_viability*labeled_ideas.labels_feasibility*labeled_ideas.labels_desirability

    multiply_all_df = labeled_ideas.nlargest(10, 'multiply_all', keep='all')
    sum_all_df = labeled_ideas.nlargest(10, 'sum_all',keep='all')
    sum_3_df = labeled_ideas.nlargest(10, 'sum_3',keep='all')
    sum_ground_truth=labeled_ideas.nlargest(10, 'ground_truth',keep='all')
    list1=multiply_all_df.Id.tolist()
    list2=sum_ground_truth.Id.tolist()
    print('multiply_all_df ')
    print(multiply_all_df)
    print(sum_ground_truth)
    print("RBOOOOO")
    print(list1, list2)
    final_rbo=RBO(list1, list2, p=0.98)
    print(final_rbo)
    return final_rbo



def hitk(classifier_desirability, classifier_feasibility, classifier_viability, classifier_overall,groundtruth_2,classifier):
    classifier_init = classifier_method()

    groundtruth_2['idea'] = groundtruth_2['idea'].apply(classifier_init.clean_text)
    # print(groundtruth_2['winning'])
    total_winning=groundtruth_2[groundtruth_2['winning'] == 1].count()["winning"]
    if classifier=='bert_based':
        groundtruth_2['idea'] = prepare_bert(groundtruth_2['idea'])

    groundtruth_2['desirability'] =classifier_desirability.predict(groundtruth_2['idea'])
    groundtruth_2['feasibility'] = classifier_feasibility.predict(groundtruth_2['idea'])
    groundtruth_2['viability'] = classifier_viability.predict(groundtruth_2['idea'])
    groundtruth_2['overall'] = classifier_overall.predict(groundtruth_2['idea'])
    groundtruth_2['multiply_all'] = groundtruth_2.desirability*groundtruth_2.feasibility*groundtruth_2.viability*groundtruth_2.overall
    groundtruth_2['sum_all'] = groundtruth_2.desirability+groundtruth_2.feasibility+groundtruth_2.viability+groundtruth_2.overall
    groundtruth_2['sum_3'] = groundtruth_2.desirability+groundtruth_2.feasibility+groundtruth_2.viability

    multiply_all_df=groundtruth_2.nlargest(20, 'multiply_all')
    sum_all_df = groundtruth_2.nlargest(20, 'sum_all')
    sum_3_df = groundtruth_2.nlargest(20, 'sum_3')
    print(sum_all_df)
    print("###############")
    print(multiply_all_df[multiply_all_df['winning'] == 1].count()["winning"])
    print(total_winning)


    multiply_all_winning = (multiply_all_df[multiply_all_df['winning'] == 1].count()["winning"])/(total_winning)
    sum_all_winning = (sum_all_df[sum_all_df['winning'] == 1].count()["winning"]) / (total_winning)
    sum_3_winning = (sum_3_df[sum_3_df['winning'] == 1].count()["winning"])/ (total_winning)

    desirability_df = groundtruth_2.nlargest(20, 'desirability')

    feasibility_df = groundtruth_2.nlargest(20, 'feasibility')
    viability_df = groundtruth_2.nlargest(20, 'viability')
    overall_df = groundtruth_2.nlargest(20, 'overall')

    desirability_winning = (desirability_df[desirability_df['winning'] == 1].count()["winning"]) /(total_winning)
    feasibility_winning =(feasibility_df[feasibility_df['winning'] == 1].count()["winning"] )/ (total_winning)
    viability_winning = (viability_df[viability_df['winning'] == 1].count()["winning"])/ (total_winning)
    overall_winning = (overall_df[overall_df['winning'] == 1].count()["winning"]) / (total_winning)
    print("###############")
    print(multiply_all_winning,sum_all_winning,sum_3_winning,desirability_winning,feasibility_winning,viability_winning,overall_winning)

    return multiply_all_winning,sum_all_winning,sum_3_winning,desirability_winning,feasibility_winning,viability_winning,overall_winning










def prepare_bert(input_X):
    model_class, tokenizer_class, pretrained_weights = (
        ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')


    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    tokenized = input_X.apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=500, truncation=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    input_X = last_hidden_states[0][:, 0, :].numpy()
    return input_X



def e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,idea_dictionnary):
    # start E step
    # updating z_i
    #print(mu)
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
            #print(answers_i[answers_i['worker'] == j].iloc[0, 2], (Aj[j_index, 0] / Bj[j_index, 0]), mj[j_index], W,V)

        W = W + (mu[i_index] / sigma_sqr[i_index, 0])
        V = V + (1 / sigma_sqr[i_index])

        mu[i_index] = W/V
        #print(W, V, W / V,mu[i_index], W/V)
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
        if Bj[j_index] == 0:
            Bj[j_index] = 0.5
        K = (ideas_j.shape[0] * (Aj[j_index] / Bj[j_index])) + alphaj[j_index]
        # print("k",K)
        L = 0
        for i in ideas_j:
            i_index = idea_dictionnary[i]
            L = L + (answers_j[answers_j['idea'] == i].iloc[0, 2] - mu[i_index])
        L = (Aj[j_index] / Bj[j_index]) * L
        alphaj[j_index] = 1 / K
        mj[j_index] = L / K
    #print('endd',mu)
    return mu, sigma_sqr, Aj, Bj, alphaj, mj


def m_step(input_X, Y_train, mu, classifier_chosen):
    # start M step
    #print('startt',mu)
    #print(mu[Y_train.shape[0]:])
    lowerBound, upperBound = 1, 5
    mu = np.rint(np.array(mu))
    mu = np.clip(mu, lowerBound, upperBound, out=mu)
    #prob_e_step = np.where(np.append(Y_train, mu[Y_train.shape[0]:]) > 0.5,0, 1)
    #print('starttprob_e_step', prob_e_step)
    #print(Y_train)
    prob_e_step = np.append(Y_train, mu[Y_train.shape[0]:])
    #print('prob_e_step',prob_e_step[Y_train.shape[0]:Y_train.shape[0]+y_val.shape[0]])
    #print(prob_e_step)
    #input_X_temp = np.concatenate((input_X[:Y_train.shape[0]], input_X[Y_train.shape[0]:Y_train.shape[0]+y_val.shape[0]]))
    # print("prob_e_steppppppppppp")
    # print(len(input_X_temp),len(prob_e_step))
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
    parser.add_argument("--groundtruth_2",
                        type=str,
                        required=True,
                        help="groundtruth_2")
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
                        choices=['svm', 'logreg', 'logreg_emb','bert_based','orderedlog','randreg'],
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


def main(labeled_ideas,answer_matrix,evaluation_file,iterr,sup_rate,classifier,indices_train,indices_test,answer_matrix_unlabeled,X_unlabeled,max_budget,target_Y,unlabeled_criteria,precision_overall_before,precision_overall_after,recall_overall_before,recall_overall_after,accuracy_overall_before,accuracy_overall_after,type_y,input_id ):

    #answer_matrix['rating'] = (answer_matrix['rating'] - 1) / 4
    #answer_matrix_unlabeled['rating']=(answer_matrix_unlabeled['rating']-1)/4
    # answer_matrix['rating'] = (answer_matrix['rating'] - 1) / 4
    # answer_matrix_unlabeled['rating']=(answer_matrix_unlabeled['rating']-1)/4

    n_ideas = labeled_ideas.shape[0]
    n_workers = answer_matrix.worker.unique().size

    iterr = iterr

    # initializing the parameters
    sigma_sqr, A, B, alpha = init_probabilities(n_ideas)
    Aj, Bj, alphaj, mj = init_Aj_Bj(A, B, n_workers, alpha)

    worker_dictionnary = dict(
        zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
    idea_dictionnary = dict(
        zip(answer_matrix['idea'].unique(), np.arange(answer_matrix['idea'].unique().shape[0])))

    # initializing the classification model and cleaning the idea text
    classifier_init = classifier_method()
    # clean the text
    # input_X = labeled_ideas['idea'].apply(classifier_init.clean_text)
    input_X = labeled_ideas

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
    elif choice_classifier == 'randreg':
        classifier_chosen = classifier_init.regressor()
        classifier_chosen.fit(input_X[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(input_X[indices_test])
    elif choice_classifier == 'orderedlog':
        classifier_chosen = classifier_init.orderedlog_review_main()
        classifier_chosen.fit(input_X[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(input_X[indices_test])
    elif choice_classifier == 'logreg_emb':
        classifier_chosen, X_train_word_average, X_test_word_average = classifier_init.logreg_embedding(
            input_X[indices_train], input_X[indices_test])
        classifier_chosen = classifier_chosen.fit(X_train_word_average, target_Y[indices_train])
        Y_pred = classifier_chosen.predict(X_test_word_average)
        input_X = np.concatenate((X_train_word_average, X_test_word_average), axis=0)
    elif choice_classifier == 'bert_based':

        input_X, classifier_chosen = classifier_init.bert_based(input_X)

        classifier_chosen.fit(input_X[indices_train], target_Y[indices_train])
        Y_pred = classifier_chosen.predict(input_X[indices_test])
    from math import sqrt
    MSE = mean_squared_error(target_Y[indices_test].values, Y_pred)
    print(MSE)
    RMSE = sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred))

    MSE_before = mean_squared_error(target_Y[indices_test].values, Y_pred)
    # print(MSE)
    RMSE_before = sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred))

    Y_pred_test = classifier_chosen.predict(X_test)
    MSE_test_before = mean_squared_error(target_Y[indices_test].values, Y_pred)
    # print(MSE)
    RMSE_test_before = sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred))

    print(RMSE)
    with open(evaluation_file, 'a') as f:
        f.write(str(sup_rate) + '_' + type_y + '_before_MSE %s' % MSE + '\n')
        f.write(str(sup_rate) + '_' + type_y + '_before_RMSE %s' % RMSE + '\n')
    # Calculate the absolute errors
    errors = abs(Y_pred - target_Y[indices_test].values)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / target_Y[indices_test].values)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    lowerBound, upperBound = 1, 5
    Y_pred = np.rint(np.array(Y_pred))
    Y_pred = np.clip(Y_pred, lowerBound, upperBound, out=Y_pred)

    print(classification_report(target_Y[indices_test], Y_pred))
    # print(Y_pred)
    # print(precision_score(target_Y[indices_test], Y_pred))

    # reporting the results
    # report = classification_report(target_Y[indices_test], Y_pred)
    # classifier_init.classification_report_csv(report, evaluation_file)
    # precision_overall_before.append(precision_score(target_Y[indices_test], Y_pred))
    # recall_overall_before.append(recall_score(target_Y[indices_test], Y_pred))
    # accuracy_overall_before.append(f1_score(target_Y[indices_test], Y_pred))
    # with open(evaluation_file, 'a') as f:
    #     f.write(str(sup_rate) + '_' + type_y + '_before_precision %s' % precision_score(target_Y[indices_test], Y_pred,
    #                                                                                     average='macro') + '\n')
    #     f.write(str(sup_rate) + '_' + type_y + '_before_recall %s' % recall_score(target_Y[indices_test], Y_pred,
    #                                                                               average='macro') + '\n')
    #     f.write(str(sup_rate) + '_' + type_y + '_before_fscore %s' % f1_score(target_Y[indices_test], Y_pred,
    #                                                                           average='macro') + '\n')

    # print(target_Y[indices_train])

    mu = np.append(target_Y[indices_train], Y_pred)

    mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj, classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj,
                                                                           sigma_sqr, alphaj,
                                                                           mj, input_X, target_Y[indices_train], mu,
                                                                           classifier_chosen, iterr, idea_dictionnary,
                                                                           evaluation_file)

    Y_pred = classifier_chosen.predict(input_X[indices_test])
    # print("target_Y[indices_test]",target_Y[indices_test])
    # print("Y_pred",Y_pred)

    MSE_after = mean_squared_error(target_Y[indices_test].values, Y_pred)
    # print(MSE)
    RMSE_after = sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred))

    Y_pred_test = classifier_chosen.predict(X_test)
    MSE_test_after = mean_squared_error(target_Y[indices_test].values, Y_pred)
    # print(MSE)
    RMSE_test_after = sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred))
    with open(evaluation_file, 'a') as f:
        f.write(
            str(sup_rate) + '_' + type_y + '_MSE %s' % mean_squared_error(target_Y[indices_test].values, Y_pred) + '\n')
        f.write(str(sup_rate) + '_' + type_y + '_RMSE %s' %
                sqrt(mean_squared_error(target_Y[indices_test].values, Y_pred)) + '\n')
    # lowerBound, upperBound = 1, 5
    # Y_pred = np.rint(np.array(Y_pred))
    # Y_pred = np.clip(Y_pred, lowerBound, upperBound, out=Y_pred)
    #
    # print(classification_report(target_Y[indices_test], Y_pred))
    # # precision_overall_after.append(precision_score(target_Y[indices_test], Y_pred))
    # # recall_overall_after.append(recall_score(target_Y[indices_test], Y_pred))
    # # accuracy_overall_after.append(f1_score(target_Y[indices_test], Y_pred))
    #
    # with open(evaluation_file, 'a') as f:
    #     f.write(str(sup_rate) + '_' + type_y + '_precision %s' % precision_score(target_Y[indices_test], Y_pred,
    #                                                                              average='macro') + '\n')
    #     f.write(str(sup_rate) + '_' + type_y + '_recall %s' % recall_score(target_Y[indices_test], Y_pred,
    #                                                                        average='macro') + '\n')
    #     f.write(str(sup_rate) + '_' + type_y + '_fscore %s' % f1_score(target_Y[indices_test], Y_pred,
    #                                                                    average='macro') + '\n')
    # # lowerBound, upperBound = 1, 5
    # Y_pred = np.rint(np.array(Y_pred))
    # Y_pred = np.clip(Y_pred, lowerBound, upperBound, out=Y_pred)


    # print(classification_report(target_Y[indices_test], Y_pred))
    # precision_overall_after.append(precision_score(target_Y[indices_test], Y_pred))
    # recall_overall_after.append(recall_score(target_Y[indices_test], Y_pred))
    # accuracy_overall_after.append(f1_score(target_Y[indices_test], Y_pred))


    # with open(evaluation_file, 'a') as f:
    #     f.write(str(sup_rate)+'_'+type_y+'_precision %s' % precision_score(target_Y[indices_test], Y_pred,average='macro') + '\n')
    #     f.write(str(sup_rate)+'_'+type_y+'_recall %s' % recall_score(target_Y[indices_test], Y_pred,average='macro') + '\n')
    #     f.write(str(sup_rate)+'_'+type_y+'_fscore %s' % f1_score(target_Y[indices_test], Y_pred,average='macro') + '\n')

    # ##active learning
    # X_unlabeled_id = X_unlabeled['Id']
    #
    # X_unlabeled = X_unlabeled['idea'].apply(classifier_init.clean_text)
    # print(len(X_unlabeled))
    #
    # print("#########beginning of active learning")
    # AL = ActiveLearner(strategy='entropy')
    # budget=0
    # X_train = input_X[indices_train]
    #
    # while budget < max_budget:
    #     budget=budget+1
    #     query_index=AL.rank(classifier_chosen, X_unlabeled, num_queries=1)
    #     print("query_index",query_index)
    #
    #     X_train=np.append(X_train,X_unlabeled[query_index].values)
    #     input_X=np.append(X_train,X_val)
    #     mu = np.append(mu[:y_train.shape[0]], unlabeled_criteria[query_index])
    #     y_train = np.append(y_train, unlabeled_criteria[query_index])
    #
    #     mu = np.append(mu, Y_pred)
    #
    #     for indx in query_index:
    #
    #         temp_answer_matrix=answer_matrix_unlabeled[answer_matrix_unlabeled['idea'] == X_unlabeled_id[indx]]
    #         answer_matrix = answer_matrix.append(temp_answer_matrix,ignore_index = True)
    #         # X_unlabeled_id = X_unlabeled_id.drop([X_unlabeled_id.index[indx]])
    #         # X_unlabeled= X_unlabeled.drop([X_unlabeled.index[indx]])
    #         # unlabeled_criteria=unlabeled_criteria.drop([unlabeled_criteria.index[indx]])
    #     sigma_sqr= np.vstack([sigma_sqr, [sigma_sqr.mean()]])
    #
    #     worker_dictionnary = dict(
    #         zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
    #     idea_dictionnary = dict(
    #         zip(answer_matrix['idea'].unique(), np.arange(answer_matrix['idea'].unique().shape[0])))
    #
    #
    #
    #     mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj, classifier_chosen = var_em(answer_matrix, worker_dictionnary, Aj, Bj,
    #                                                                            sigma_sqr, alphaj,
    #                                                                            mj, input_X, y_train, mu,
    #                                                                            classifier_chosen, iterr, idea_dictionnary,
    #                                                                            target_Y)
    #     predict_theta_i = classifier_chosen.predict(X_val)
    #     print(classification_report(target_Y[indices_test], predict_theta_i))
    #
    #     report = classification_report(target_Y[indices_test], predict_theta_i)
    #     classifier_init.classification_report_csv(report, evaluation_file)
    #     with open(evaluation_file, 'a') as f:
    #         f.write('accuracy %s' % f1_score(target_Y[indices_test], predict_theta_i))
    return classifier_chosen,input_X[indices_test],target_Y[indices_test],MSE_before, RMSE_before, MSE_test_before, RMSE_test_before,MSE_after,RMSE_after,MSE_test_after,RMSE_test_after


if __name__ == '__main__':
  precision_desirability_before=[]
  precision_desirability_after = []
  recall_desirability_before = []
  recall_desirability_after = []
  accuracy_desirability_before = []
  accuracy_desirability_after = []

  precision_viability_before = []
  precision_viability_after = []
  recall_viability_before = []
  recall_viability_after = []
  accuracy_viability_before = []
  accuracy_viability_after = []

  precision_feasibility_before = []
  precision_feasibility_after = []
  recall_feasibility_before = []
  recall_feasibility_after = []
  accuracy_feasibility_before = []
  accuracy_feasibility_after = []

  precision_overall_before = []
  precision_overall_after = []
  recall_overall_before = []
  recall_overall_after = []
  accuracy_overall_before = []
  accuracy_overall_after = []



  precision_combined=[]
  recall_combined=[]
  accuracy_combined=[]

  precision_combined_2 = []
  recall_combined_2 = []
  accuracy_combined_2 = []
  args_main = parse_args()
  max_budget = args_main.max_budget
  iterr = args_main.iterr
  # sup_rate=args_main.sup_rate
  classifier = args_main.classifier
  # initializing the classification model and cleaning the idea text
  classifier_init = classifier_method()
  # clean the text


  sup_rate_list=[1,2,3,4,5,6,7,8,9]
  for sup_rate in sup_rate_list:
        print('#############started......',sup_rate)
        sup_rate = 0.1 * sup_rate
        MSE_before_desirabilitylist = []
        RMSE_before_desirabilitylist = []
        MSE_test_before_desirabilitylist = []
        RMSE_test_before_desirabilitylist = []
        MSE_after_desirabilitylist = []
        RMSE_after_desirabilitylist = []
        MSE_test_after_desirabilitylist = []
        RMSE_test_after_desirabilitylist = []

        MSE_before_viabilitylist = []
        RMSE_before_viabilitylist = []
        MSE_test_before_viabilitylist = []
        RMSE_test_before_viabilitylist = []
        MSE_after_viabilitylist = []
        RMSE_after_viabilitylist = []
        MSE_test_after_viabilitylist = []
        RMSE_test_after_viabilitylist = []

        MSE_before_feasibilitylist = []
        RMSE_before_feasibilitylist = []
        MSE_test_before_feasibilitylist = []
        RMSE_test_before_feasibilitylist = []
        MSE_after_feasibilitylist = []
        RMSE_after_feasibilitylist = []
        MSE_test_after_feasibilitylist = []
        RMSE_test_after_feasibilitylist = []

        MSE_before_overalllist = []
        RMSE_before_overalllist = []
        MSE_test_before_overalllist = []
        RMSE_test_before_overalllist = []
        MSE_after_overalllist = []
        RMSE_after_overalllist = []
        MSE_test_after_overalllist = []
        RMSE_test_after_overalllist = []

        multiply_all_winning_list=[]
        sum_all_winning_list=[]
        sum_3_winning_list=[]
        desirability_winning_list=[]
        feasibility_winning_list=[]
        viability_winning_list=[]
        overall_winning_list=[]
        rbo_list=[]

        for iteration in range(0,9):
            labeled_ideas = pd.read_csv(
                args_main.inputfile_ideas)
            labeled_ideas = labeled_ideas.sample(frac=1).reset_index(drop=True)
            input_X = labeled_ideas['idea'].apply(classifier_init.clean_text)
            input_id = labeled_ideas['Id']
            if classifier == 'bert_based':

                input_X = prepare_bert(input_X)


            X_unlabeled = pd.read_csv(
                args_main.unlabeled_ideas)
            answer_matrix_viability = pd.read_csv(
                args_main.answer_matrix_viability)
            groundtruth_2=pd.read_csv(
                args_main.groundtruth_2)
            # print(answer_matrix_viability['rating'].value_counts())
            answer_matrix_desirability = pd.read_csv(
                args_main.answer_matrix_desirability)
            # print(answer_matrix_desirability['rating'].value_counts())
            answer_matrix_feasibility = pd.read_csv(
                args_main.answer_matrix_feasibility)
            # print(answer_matrix_feasibility['rating'].value_counts())
            answer_matrix_overall = pd.read_csv(
                 args_main.answer_matrix)
            # print(answer_matrix_overall['rating'].value_counts())


            answer_matrix_unlabeled_feasibility= pd.read_csv(
                args_main.answer_matrix_unlabeled_feasibility)
            answer_matrix_unlabeled_desirability = pd.read_csv(
                args_main.answer_matrix_unlabeled_desirability)
            answer_matrix_unlabeled_viability = pd.read_csv(
                args_main.answer_matrix_unlabeled_viability)
            answer_matrix_unlabeled_overall = pd.read_csv(
                args_main.answer_matrix_unlabeled)
            evaluation_file_viability= args_main.evaluation_file_viability
            evaluation_file_desirability= args_main.evaluation_file_desirability
            evaluation_file_feasibility = args_main.evaluation_file_feasibility
            evaluation_file_overall = args_main.evaluation_file

            labels_list = np.char.mod('%d', labeled_ideas['labels'].unique()).tolist()

            target_Y_viability = labeled_ideas['labels_viability']
            target_Y_feasibility = labeled_ideas['labels_feasibility']
            target_Y_desirability = labeled_ideas['labels_desirability']
            target_Y_overall = labeled_ideas['labels']

            unlabeled_viability = X_unlabeled['labels_viability']
            unlabeled_feasibility = X_unlabeled['labels_feasibility']
            unlabeled_desirability = X_unlabeled['labels_desirability']
            unlabeled_overall = X_unlabeled['labels']
            target_Y_winner = labeled_ideas['rating']
            target_Y_final_feedback = labeled_ideas['final_feedback']



            indices=[]
            for ind in range(0, len(target_Y_overall)):
                indices.append(ind)
            # print(len(target_Y_overall))
            # print(indices)

            # splitting the data
            X_train, X_test, Y_train, Y_test,indices_train,indices_test = train_test_split(input_X, target_Y_overall,indices,
                                                                test_size=(1-sup_rate),shuffle=False)

        # try:


            classifier_desirability,test_desirability,y_desirability,MSE_before, RMSE_before, MSE_test_before, RMSE_test_before,MSE_after,RMSE_after,MSE_test_after,RMSE_test_after= main(input_X, answer_matrix_desirability, evaluation_file_desirability,
                                                       iterr, sup_rate,
                                                       classifier, indices_train, indices_test,
                                                       answer_matrix_unlabeled_desirability, X_unlabeled, max_budget,
                                                       target_Y_desirability,unlabeled_desirability,precision_desirability_before,precision_desirability_after,recall_desirability_before,recall_desirability_after,accuracy_desirability_before,accuracy_desirability_after,'normal' ,input_id )

            MSE_before_desirabilitylist.append(MSE_before)
            RMSE_before_desirabilitylist.append(RMSE_before)
            MSE_test_before_desirabilitylist.append(MSE_test_before)
            RMSE_test_before_desirabilitylist.append(RMSE_test_before)
            MSE_after_desirabilitylist.append(MSE_after)
            RMSE_after_desirabilitylist.append(RMSE_after)
            MSE_test_after_desirabilitylist.append(MSE_test_after)
            RMSE_test_after_desirabilitylist.append(RMSE_test_after)

            classifier_viability,test_viability,y_viability,MSE_before, RMSE_before, MSE_test_before, RMSE_test_before,MSE_after,RMSE_after,MSE_test_after,RMSE_test_after = main(input_X, answer_matrix_viability, evaluation_file_viability, iterr,
                                                 sup_rate, classifier, indices_train, indices_test,
                                                 answer_matrix_unlabeled_viability, X_unlabeled, max_budget, target_Y_viability,unlabeled_viability,precision_viability_before,precision_viability_after,recall_viability_before,recall_viability_after,accuracy_viability_before,accuracy_viability_after ,'normal',input_id )

            MSE_before_viabilitylist.append(MSE_before)
            RMSE_before_viabilitylist.append(RMSE_before)
            MSE_test_before_viabilitylist.append(MSE_test_before)
            RMSE_test_before_viabilitylist.append(RMSE_test_before)
            MSE_after_viabilitylist.append(MSE_after)
            RMSE_after_viabilitylist.append(RMSE_after)
            MSE_test_after_viabilitylist.append(MSE_test_after)
            RMSE_test_after_viabilitylist.append(RMSE_test_after)



            classifier_feasibility,test_feasibility,y_feasibility,MSE_before, RMSE_before, MSE_test_before, RMSE_test_before,MSE_after,RMSE_after,MSE_test_after,RMSE_test_after=main(input_X, answer_matrix_feasibility, evaluation_file_feasibility, iterr, sup_rate,
                                           classifier,indices_train,indices_test,answer_matrix_unlabeled_feasibility,X_unlabeled,max_budget,target_Y_feasibility,unlabeled_feasibility,precision_feasibility_before,precision_feasibility_after,recall_feasibility_before,recall_feasibility_after,accuracy_feasibility_before,accuracy_feasibility_after ,'normal',input_id )

            MSE_before_feasibilitylist.append(MSE_before)
            RMSE_before_feasibilitylist.append(RMSE_before)
            MSE_test_before_feasibilitylist.append(MSE_test_before)
            RMSE_test_before_feasibilitylist.append(RMSE_test_before)
            MSE_after_feasibilitylist.append(MSE_after)
            RMSE_after_feasibilitylist.append(RMSE_after)
            MSE_test_after_feasibilitylist.append(MSE_test_after)
            RMSE_test_after_feasibilitylist.append(RMSE_test_after)



            classifier_overall,test_overall,y_overall,MSE_before, RMSE_before, MSE_test_before, RMSE_test_before,MSE_after,RMSE_after,MSE_test_after,RMSE_test_after=main(input_X, answer_matrix_overall, evaluation_file_overall, iterr, sup_rate,
                                           classifier,indices_train,indices_test,answer_matrix_unlabeled_overall,X_unlabeled,max_budget,target_Y_overall,unlabeled_overall,precision_overall_before,precision_overall_after,recall_overall_before,recall_overall_after,accuracy_overall_before,accuracy_overall_after,'normal',input_id )

            MSE_before_overalllist.append(MSE_before)
            RMSE_before_overalllist.append(RMSE_before)
            MSE_test_before_overalllist.append(MSE_test_before)
            RMSE_test_before_overalllist.append(RMSE_test_before)
            MSE_after_overalllist.append(MSE_after)
            RMSE_after_overalllist.append(RMSE_after)
            MSE_test_after_overalllist.append(MSE_test_after)
            RMSE_test_after_overalllist.append(RMSE_test_after)

            # multiply_all_winning, sum_all_winning, sum_3_winning, desirability_winning, feasibility_winning, viability_winning, overall_winning=hitk(classifier_desirability, classifier_feasibility, classifier_viability, classifier_overall,groundtruth_2,classifier)
            multiply_all_winning, sum_all_winning, sum_3_winning, desirability_winning, feasibility_winning, viability_winning, overall_winning = 0,0,0,0,0,0,0

            final_rbo=measure_rbo(classifier_desirability, classifier_feasibility, classifier_viability, classifier_overall,
                        labeled_ideas,indices_test, classifier)
            rbo_list.append(final_rbo)


            with open('../output/evaluation_file_g2.csv', 'a') as f:
                f.write(str(sup_rate) + '_multiply_all_winning %s' % multiply_all_winning + '\n')
                f.write(str(sup_rate) + '_sum_all_winning %s' % sum_all_winning + '\n')
                f.write(str(sup_rate) + '_sum_3_winning %s' % sum_3_winning + '\n')
                f.write(str(sup_rate) + '_desirability_winning %s' % desirability_winning + '\n')
                f.write(str(sup_rate) + '_feasibility_winning %s' % feasibility_winning + '\n')
                f.write(str(sup_rate) + '_viability_winning %s' % viability_winning + '\n')
                f.write(str(sup_rate) + '_overall_winning %s' % overall_winning + '\n')

            multiply_all_winning_list.append(multiply_all_winning)
            sum_all_winning_list.append(sum_all_winning)
            sum_3_winning_list.append(sum_3_winning)
            desirability_winning_list.append(desirability_winning)
            feasibility_winning_list.append(feasibility_winning)
            viability_winning_list.append(viability_winning)
            overall_winning_list.append(overall_winning)



            # #####################
            # try:
            #
            #
            #     classifier_desirability_winner,test_desirability,y_desirability,precision_desirability_before,precision_desirability_after,recall_desirability_before,recall_desirability_after,accuracy_desirability_before,accuracy_desirability_after = main(input_X, answer_matrix_desirability, evaluation_file_desirability,
            #                                                iterr, sup_rate,
            #                                                classifier, indices_train, indices_test,
            #                                                answer_matrix_unlabeled_desirability, X_unlabeled, max_budget,
            #                                                target_Y_winner,unlabeled_desirability,precision_desirability_before,precision_desirability_after,recall_desirability_before,recall_desirability_after,accuracy_desirability_before,accuracy_desirability_after,'winner' )
            #     classifier_viability_winner,test_viability,y_viability,precision_viability_before,precision_viability_after,recall_viability_before,recall_viability_after,accuracy_viability_before,accuracy_viability_after = main(input_X, answer_matrix_viability, evaluation_file_viability, iterr,
            #                                          sup_rate, classifier, indices_train, indices_test,
            #                                          answer_matrix_unlabeled_viability, X_unlabeled, max_budget, target_Y_winner,unlabeled_viability,precision_viability_before,precision_viability_after,recall_viability_before,recall_viability_after,accuracy_viability_before,accuracy_viability_after,'winner' )
            #
            #     classifier_feasibility_winner,test_feasibility,y_feasibility,precision_feasibility_before,precision_feasibility_after,recall_feasibility_before,recall_feasibility_after,accuracy_feasibility_before,accuracy_feasibility_after=main(input_X, answer_matrix_feasibility, evaluation_file_feasibility, iterr, sup_rate,
            #                                    classifier,indices_train,indices_test,answer_matrix_unlabeled_feasibility,X_unlabeled,max_budget,target_Y_winner,unlabeled_feasibility,precision_feasibility_before,precision_feasibility_after,recall_feasibility_before,recall_feasibility_after,accuracy_feasibility_before,accuracy_feasibility_after,'winner' )
            #
            #
            #
            #
            #
            #
            #     classifier_overall_winner,test_overall,y_overall,precision_overall_before,precision_overall_after,recall_overall_before,recall_overall_after,accuracy_overall_before,accuracy_overall_after=main(input_X, answer_matrix_overall, evaluation_file_overall, iterr, sup_rate,
            #                                    classifier,indices_train,indices_test,answer_matrix_unlabeled_overall,X_unlabeled,max_budget,target_Y_winner,unlabeled_overall,precision_overall_before,precision_overall_after,recall_overall_before,recall_overall_after,accuracy_overall_before,accuracy_overall_after,'winner' )
            #     predict_desirability_winner = classifier_desirability_winner.predict(test_overall)
            #     predict_viability_winner = classifier_viability_winner.predict(test_overall)
            #     predict_feasibility_winner = classifier_feasibility_winner.predict(test_overall)
            #     predict_overall_winner = classifier_overall_winner.predict(test_overall)
            #
            #
            #     #
            #     # lowerBound, upperBound = 1, 5
            #     # predict_desirability_winner= np.rint(np.array(predict_desirability_winner))
            #     # predict_desirability_winner= np.clip(predict_desirability_winner , lowerBound, upperBound, out=predict_desirability_winner)
            #     # predict_viability_winner= np.rint(np.array(predict_viability_winner))
            #     # predict_viability_winner= np.clip(predict_viability_winner, lowerBound, upperBound, out=predict_viability_winner)
            #     # predict_feasibility_winner= np.rint(np.array(predict_feasibility_winner))
            #     # predict_feasibility_winner= np.clip(predict_feasibility_winner, lowerBound, upperBound, out=predict_feasibility_winner)
            #     # predict_overall_winner= np.rint(np.array(predict_overall_winner))
            #     # predict_overall_winner= np.clip(predict_overall_winner, lowerBound, upperBound, out=predict_overall_winner)
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(
            #             str(sup_rate) + '_precisionwinner_desirability %s' % precision_score(target_Y_winner[indices_test],
            #                                                                                  predict_desirability) + '\n')
            #         f.write(str(sup_rate) + '_recallwinner_desirability %s' % recall_score(target_Y_winner[indices_test],
            #                                                                                predict_desirability) + '\n')
            #         f.write(str(sup_rate) + '_fscorewinner_desirability %s' % f1_score(target_Y_winner[indices_test],
            #                                                                            predict_desirability) + '\n')
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionwinner_viability %s' % precision_score(target_Y_winner[indices_test],
            #                                                                                   predict_viability) + '\n')
            #         f.write(str(sup_rate) + '_recallwinner_viability %s' % recall_score(target_Y_winner[indices_test],
            #                                                                             predict_viability) + '\n')
            #         f.write(str(sup_rate) + '_fscorewinner_viability %s' % f1_score(target_Y_winner[indices_test],
            #                                                                         predict_viability) + '\n')
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(
            #             str(sup_rate) + '_precisionwinner_feasibility %s' % precision_score(target_Y_winner[indices_test],
            #                                                                                 predict_feasibility) + '\n')
            #         f.write(str(sup_rate) + '_recallwinner_feasibility %s' % recall_score(target_Y_winner[indices_test],
            #                                                                               predict_feasibility) + '\n')
            #         f.write(str(sup_rate) + '_fscorewinner_feasibility %s' % f1_score(target_Y_winner[indices_test],
            #                                                                           predict_feasibility) + '\n')
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionwinner_overall %s' % precision_score(target_Y_winner[indices_test],
            #                                                                                 predict_overall) + '\n')
            #         f.write(str(sup_rate) + '_recallwinner_overall %s' % recall_score(target_Y_winner[indices_test],
            #                                                                           predict_overall) + '\n')
            #         f.write(str(sup_rate) + '_fscorewinner_overall %s' % f1_score(target_Y_winner[indices_test],
            #                                                                       predict_overall) + '\n')
            # except:
            #     pass
            # try:
            #
            #     classifier_desirability_feedback, test_desirability, y_desirability, precision_desirability_before, precision_desirability_after, recall_desirability_before, recall_desirability_after, accuracy_desirability_before, accuracy_desirability_after = main(
            #         input_X, answer_matrix_desirability, evaluation_file_desirability,
            #         iterr, sup_rate,
            #         classifier, indices_train, indices_test,
            #         answer_matrix_unlabeled_desirability, X_unlabeled, max_budget,
            #         target_Y_final_feedback, unlabeled_desirability, precision_desirability_before, precision_desirability_after,
            #         recall_desirability_before, recall_desirability_after, accuracy_desirability_before,
            #         accuracy_desirability_after,'feedback')
            #     classifier_viability_feedback, test_viability, y_viability, precision_viability_before, precision_viability_after, recall_viability_before, recall_viability_after, accuracy_viability_before, accuracy_viability_after = main(
            #         input_X, answer_matrix_viability, evaluation_file_viability, iterr,
            #         sup_rate, classifier, indices_train, indices_test,
            #         answer_matrix_unlabeled_viability, X_unlabeled, max_budget, target_Y_final_feedback, unlabeled_viability,
            #         precision_viability_before, precision_viability_after, recall_viability_before, recall_viability_after,
            #         accuracy_viability_before, accuracy_viability_after,'feedback')
            #
            #     classifier_feasibility_feedback, test_feasibility, y_feasibility, precision_feasibility_before, precision_feasibility_after, recall_feasibility_before, recall_feasibility_after, accuracy_feasibility_before, accuracy_feasibility_after = main(
            #         input_X, answer_matrix_feasibility, evaluation_file_feasibility, iterr, sup_rate,
            #         classifier, indices_train, indices_test, answer_matrix_unlabeled_feasibility, X_unlabeled, max_budget,
            #         target_Y_final_feedback, unlabeled_feasibility, precision_feasibility_before, precision_feasibility_after,
            #         recall_feasibility_before, recall_feasibility_after, accuracy_feasibility_before,
            #         accuracy_feasibility_after,'feedback')
            #
            #     classifier_overall_feedback, test_overall, y_overall, precision_overall_before, precision_overall_after, recall_overall_before, recall_overall_after, accuracy_overall_before, accuracy_overall_after = main(
            #         input_X, answer_matrix_overall, evaluation_file_overall, iterr, sup_rate,
            #         classifier, indices_train, indices_test, answer_matrix_unlabeled_overall, X_unlabeled, max_budget,
            #         target_Y_final_feedback, unlabeled_overall, precision_overall_before, precision_overall_after,
            #         recall_overall_before, recall_overall_after, accuracy_overall_before, accuracy_overall_after,'feedback')
            #     predict_desirability_feedback = classifier_desirability_feedback.predict(test_overall)
            #     predict_viability_feedback = classifier_viability_feedback.predict(test_overall)
            #     predict_feasibility_feedback = classifier_feasibility_feedback.predict(test_overall)
            #     predict_overall_feedback = classifier_overall_feedback.predict(test_overall)
            #     #
            #     # lowerBound, upperBound = 1, 5
            #     # predict_desirability_feedback = np.rint(np.array(predict_desirability_feedback))
            #     # predict_desirability_feedback = np.clip(predict_desirability_feedback, lowerBound, upperBound, out=predict_desirability_feedback)
            #     # predict_viability_feedback = np.rint(np.array(predict_viability_feedback))
            #     # predict_viability_feedback = np.clip(predict_viability_feedback, lowerBound, upperBound, out=predict_viability_feedback)
            #     # predict_feasibility_feedback = np.rint(np.array(predict_feasibility_feedback))
            #     # predict_feasibility_feedback = np.clip(predict_feasibility_feedback, lowerBound, upperBound, out=predict_feasibility_feedback)
            #     # predict_overall_feedback = np.rint(np.array(predict_overall_feedback))
            #     # predict_overall_feedback = np.clip(predict_overall, lowerBound, upperBound, out=predict_overall_feedback)
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionfinalfeed_desirability %s' % precision_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_desirability) + '\n')
            #         f.write(str(sup_rate) + '_recallfinalfeed_desirability %s' % recall_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_desirability) + '\n')
            #         f.write(
            #             str(sup_rate) + '_fscorefinalfeed_desirability %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                           predict_desirability) + '\n')
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionfinalfeed_viability %s' % precision_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_viability) + '\n')
            #         f.write(str(sup_rate) + '_recallfinalfeed_viability %s' % recall_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_viability) + '\n')
            #         f.write(
            #             str(sup_rate) + '_fscorefinalfeed_viability %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                        predict_viability) + '\n')
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionfinalfeed_feasibility %s' % precision_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_feasibility) + '\n')
            #         f.write(str(sup_rate) + '_recallfinalfeed_feasibility %s' % recall_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_feasibility) + '\n')
            #         f.write(
            #             str(sup_rate) + '_fscorefinalfeed_feasibility %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                          predict_feasibility) + '\n')
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionfinalfeed_overall %s' % precision_score(
            #             target_Y_final_feedback[indices_test],
            #             predict_overall) + '\n')
            #         f.write(
            #             str(sup_rate) + '_recallfinalfeed_overall %s' % recall_score(target_Y_final_feedback[indices_test],
            #                                                                          predict_overall) + '\n')
            #         f.write(str(sup_rate) + '_fscorefinalfeed_overall %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                          predict_overall) + '\n')
            #
            # except:
            #     pass

            predict_desirability = classifier_desirability.predict(test_overall)
            predict_viability = classifier_viability.predict(test_overall)
            predict_feasibility = classifier_feasibility.predict(test_overall)
            predict_overall = classifier_overall.predict(test_overall)

            lowerBound, upperBound = 1, 5
            predict_desirability = np.rint(np.array(predict_desirability))
            predict_desirability = np.clip(predict_desirability, lowerBound, upperBound, out=predict_desirability)
            predict_desirability= np.where((predict_desirability >= 4), 1., 0.)
            predict_viability = np.rint(np.array(predict_viability))
            predict_viability = np.clip(predict_viability, lowerBound, upperBound, out=predict_viability)
            predict_viability = np.where((predict_viability >= 4), 1., 0.)
            predict_feasibility = np.rint(np.array(predict_feasibility))
            predict_feasibility = np.clip(predict_feasibility, lowerBound, upperBound, out=predict_feasibility)
            predict_feasibility = np.where((predict_feasibility >= 4), 1., 0.)
            predict_overall = np.rint(np.array(predict_overall))
            predict_overall = np.clip(predict_overall, lowerBound, upperBound, out=predict_overall)
            predict_overall = np.where((predict_overall >= 4), 1., 0.)










            print("desirability",predict_desirability)
            print("viability",predict_viability)
            print("feasibility",predict_feasibility)
            print("overall",predict_overall)
            print("y_overall_truth", y_overall)
            print("y_winner",target_Y_winner[indices_test].values)
            print("y_finalfeedback", target_Y_final_feedback[indices_test].values)

            # print(np.mean([mu_feasibility, mu_viability, mu_desirability,mu_overall], axis=0))

            combined = [a * b * c*d for a, b, c,d in zip(predict_desirability, predict_viability, predict_feasibility,predict_overall)]
            print("combined",combined)
            # print(classification_report(y_overall, combined))
            # precision_combined.append(precision_score(y_overall, combined))
            # recall_combined.append(recall_score(y_overall, combined))
            # accuracy_combined.append(f1_score(y_overall, combined))

            # with open('../output/evaluation_file_combined.csv', 'a') as f:
            #     f.write(str(sup_rate) + '_precisioncombined_1 %s' % precision_score(y_overall, combined) + '\n')
            #     f.write(str(sup_rate) + '_recallcombined_1 %s' % recall_score(y_overall, combined) + '\n')
            #     f.write(str(sup_rate) + '_fscorecombined_1 %s' % f1_score(y_overall, combined) + '\n')
            # try:
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionwinner_1 %s' % precision_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #         f.write(str(sup_rate) + '_recallwinner_1 %s' % recall_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #         f.write(str(sup_rate) + '_fscorewinner_1 %s' % f1_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionfinalfeed_1 %s' % precision_score(target_Y_final_feedback[indices_test], combined,average='macro') + '\n')
            #         f.write(str(sup_rate) + '_recallfinalfeed_1 %s' % recall_score(target_Y_final_feedback[indices_test], combined,average='macro') + '\n')
            #         f.write(str(sup_rate) + '_fscorefinalfeed_1 %s' % f1_score(target_Y_final_feedback[indices_test], combined,average='macro') + '\n')
            #
            #
            #
            #     # combined = [a * b * c * d for a, b, c, d in
            #     #             zip(predict_desirability_winner, predict_viability_winner, predict_feasibility_winner, predict_overall_winner)]
            #     #
            #     # with open('../output/evaluation_file_combined.csv', 'a') as f:
            #     #     f.write(str(sup_rate) + '_precisionbothwinner_1 %s' % precision_score(target_Y_winner[indices_test], combined) + '\n')
            #     #     f.write(str(sup_rate) + '_recallbothwinner_1 %s' % recall_score(target_Y_winner[indices_test], combined) + '\n')
            #     #     f.write(str(sup_rate) + '_fscorebothwinner_1 %s' % f1_score(target_Y_winner[indices_test], combined) + '\n')
            # except:
            #     pass
            # try:
            #     combined = [a * b * c * d for a, b, c, d in
            #                 zip(predict_desirability_feedback, predict_viability_feedback, predict_feasibility_feedback,
            #                     predict_overall_feedback)]
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #             f.write(
            #                 str(sup_rate) + '_precisionbothfinalfeed_1 %s' % precision_score(target_Y_final_feedback[indices_test],
            #                                                                              combined) + '\n')
            #             f.write(str(sup_rate) + '_recallbothfinalfeed_1 %s' % recall_score(target_Y_final_feedback[indices_test],
            #                                                                            combined) + '\n')
            #             f.write(str(sup_rate) + '_fscorebothfinalfeed_1 %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                        combined) + '\n')
            #
            #
            #     combined = []
            #     for a, b, c in zip(predict_desirability_winner, predict_viability_winner, predict_feasibility_winner):
            #         finalnumber = 0
            #         if a == 1 and b == 1:
            #             finalnumber = 1
            #         elif a == 1 and c == 1:
            #             finalnumber = 1
            #         elif b == 1 and c == 1:
            #             finalnumber = 1
            #         combined.append(finalnumber)
            #
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #         f.write(str(sup_rate) + '_precisionbothwinner_2 %s' % precision_score(target_Y_winner[indices_test],
            #                                                                               combined) + '\n')
            #         f.write(
            #             str(sup_rate) + '_recallbothwinner_2 %s' % recall_score(target_Y_winner[indices_test], combined) + '\n')
            #         f.write(str(sup_rate) + '_fscorebothwinner_2 %s' % f1_score(target_Y_winner[indices_test], combined) + '\n')
            # except:
            #     pass
            # try:
            #     combined = []
            #     for a, b, c in zip(predict_desirability_feedback, predict_viability_feedback, predict_feasibility_feedback):
            #         finalnumber = 0
            #         if a == 1 and b == 1:
            #             finalnumber = 1
            #         elif a == 1 and c == 1:
            #             finalnumber = 1
            #         elif b == 1 and c == 1:
            #             finalnumber = 1
            #         combined.append(finalnumber)
            #     with open('../output/evaluation_file_combined.csv', 'a') as f:
            #             f.write(
            #                 str(sup_rate) + '_precisionbothfinalfeed_2 %s' % precision_score(target_Y_final_feedback[indices_test],
            #                                                                              combined) + '\n')
            #             f.write(str(sup_rate) + '_recallbothfinalfeed_2 %s' % recall_score(target_Y_final_feedback[indices_test],
            #                                                                            combined) + '\n')
            #             f.write(str(sup_rate) + '_fscorebothfinalfeed_2 %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                        combined) + '\n')
            # except:
            #  pass

            #
            #
            # combined = []
            # for a, b, c in zip(predict_desirability, predict_viability, predict_feasibility):
            #     finalnumber = 0
            #     if a == 1 and b == 1:
            #         finalnumber = 1
            #     elif a == 1 and c == 1:
            #         finalnumber = 1
            #     elif b == 1 and c == 1:
            #         finalnumber = 1
            #     combined.append(finalnumber)
            # print("combined 2222")
            # # print(classification_report(y_overall, combined))
            # # precision_combined_2.append(precision_score(y_overall, combined))
            # # recall_combined_2.append(recall_score(y_overall, combined))
            # # accuracy_combined_2.append(f1_score(y_overall, combined))
            # with open('../output/evaluation_file_combined.csv', 'a') as f:
            #     f.write(str(sup_rate) + '_precisionwinner_2 %s' % precision_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #     f.write(str(sup_rate) + '_recallwinner_2 %s' % recall_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #     f.write(str(sup_rate) + '_fscorewinner_2 %s' % f1_score(target_Y_winner[indices_test], combined,average='macro') + '\n')
            #
            # with open('../output/evaluation_file_combined.csv', 'a') as f:
            #     f.write(str(sup_rate) + '_precisionfinalfeed_2 %s' % precision_score(target_Y_final_feedback[indices_test],
            #                                                                          combined,average='macro') + '\n')
            #     f.write(str(sup_rate) + '_recallfinalfeed_2 %s' % recall_score(target_Y_final_feedback[indices_test],
            #                                                                    combined,average='macro') + '\n')
            #     f.write(str(sup_rate) + '_fscorefinalfeed_2 %s' % f1_score(target_Y_final_feedback[indices_test],
            #                                                                combined,average='macro') + '\n')
            # with open('../output/evaluation_file_combined.csv', 'a') as f:
            #     f.write(str(sup_rate) + '_precisioncombined_2 %s' % precision_score(y_overall, combined) + '\n')
            #     f.write(str(sup_rate) + '_recallcombined_2 %s' % recall_score(y_overall, combined) + '\n')
            #     f.write(str(sup_rate) + '_fscorecombined_2 %s' % f1_score(y_overall, combined) + '\n')

            # print("winner")
            # print(classification_report(target_Y_winner[indices_test], combined))
            # precision_combined.append(precision_score(target_Y_winner[indices_test], combined))
            # recall_combined.append(recall_score(target_Y_winner[indices_test], combined))
            # accuracy_combined.append(f1_score(target_Y_winner[indices_test], combined))

            # print("final_feedback")
            # print(classification_report(target_Y_final_feedback[indices_test], combined))
            # precision_combined.append(precision_score(target_Y_final_feedback[indices_test], combined))
            # recall_combined.append(recall_score(target_Y_final_feedback[indices_test], combined))
            # accuracy_combined.append(f1_score(target_Y_final_feedback[indices_test], combined))
        # except:
        #     pass
        MSE_before_desirabilitylist=sum(MSE_before_desirabilitylist) / len(MSE_before_desirabilitylist)
        RMSE_before_desirabilitylist=sum(RMSE_before_desirabilitylist) / len(RMSE_before_desirabilitylist)
        MSE_test_before_desirabilitylist=sum(MSE_test_before_desirabilitylist) / len(MSE_test_before_desirabilitylist)
        RMSE_test_before_desirabilitylist=sum(RMSE_test_before_desirabilitylist) / len(RMSE_test_before_desirabilitylist)
        MSE_after_desirabilitylist=sum(MSE_after_desirabilitylist) / len(MSE_after_desirabilitylist)
        RMSE_after_desirabilitylist=sum(RMSE_after_desirabilitylist) / len(RMSE_after_desirabilitylist)
        MSE_test_after_desirabilitylist=sum(MSE_test_after_desirabilitylist) / len(MSE_test_after_desirabilitylist)
        RMSE_test_after_desirabilitylist=sum(RMSE_test_after_desirabilitylist) / len(RMSE_test_after_desirabilitylist)

        MSE_before_feasibilitylist = sum(MSE_before_feasibilitylist) / len(MSE_before_feasibilitylist)
        RMSE_before_feasibilitylist = sum(RMSE_before_feasibilitylist) / len(RMSE_before_feasibilitylist)
        MSE_test_before_feasibilitylist = sum(MSE_test_before_feasibilitylist) / len(MSE_test_before_feasibilitylist)
        RMSE_test_before_feasibilitylist = sum(RMSE_test_before_feasibilitylist) / len(
            RMSE_test_before_feasibilitylist)
        MSE_after_feasibilitylist = sum(MSE_after_feasibilitylist) / len(MSE_after_feasibilitylist)
        RMSE_after_feasibilitylist = sum(RMSE_after_feasibilitylist) / len(RMSE_after_feasibilitylist)
        MSE_test_after_feasibilitylist = sum(MSE_test_after_feasibilitylist) / len(MSE_test_after_feasibilitylist)
        RMSE_test_after_feasibilitylist = sum(RMSE_test_after_feasibilitylist) / len(RMSE_test_after_feasibilitylist)

        MSE_before_viabilitylist = sum(MSE_before_viabilitylist) / len(MSE_before_viabilitylist)
        RMSE_before_viabilitylist = sum(RMSE_before_viabilitylist) / len(RMSE_before_viabilitylist)
        MSE_test_before_viabilitylist = sum(MSE_test_before_viabilitylist) / len(MSE_test_before_viabilitylist)
        RMSE_test_before_viabilitylist = sum(RMSE_test_before_viabilitylist) / len(
            RMSE_test_before_viabilitylist)
        MSE_after_viabilitylist = sum(MSE_after_viabilitylist) / len(MSE_after_viabilitylist)
        RMSE_after_viabilitylist = sum(RMSE_after_viabilitylist) / len(RMSE_after_viabilitylist)
        MSE_test_after_viabilitylist = sum(MSE_test_after_viabilitylist) / len(MSE_test_after_viabilitylist)
        RMSE_test_after_viabilitylist = sum(RMSE_test_after_viabilitylist) / len(RMSE_test_after_viabilitylist)

        MSE_before_overalllist = sum(MSE_before_overalllist) / len(MSE_before_overalllist)
        RMSE_before_overalllist = sum(RMSE_before_overalllist) / len(RMSE_before_overalllist)
        MSE_test_before_overalllist = sum(MSE_test_before_overalllist) / len(MSE_test_before_overalllist)
        RMSE_test_before_overalllist = sum(RMSE_test_before_overalllist) / len(
            RMSE_test_before_overalllist)
        MSE_after_overalllist = sum(MSE_after_overalllist) / len(MSE_after_overalllist)
        RMSE_after_overalllist = sum(RMSE_after_overalllist) / len(RMSE_after_overalllist)
        MSE_test_after_overalllist = sum(MSE_test_after_overalllist) / len(MSE_test_after_overalllist)
        RMSE_test_after_overalllist = sum(RMSE_test_after_overalllist) / len(RMSE_test_after_overalllist)

        multiply_all_winning=sum(multiply_all_winning_list)/len(multiply_all_winning_list)
        sum_all_winning=sum(sum_all_winning_list)/len(sum_all_winning_list)
        sum_3_winning=sum(sum_3_winning_list)/len(sum_3_winning_list)
        desirability_winning=sum(desirability_winning_list)/len(desirability_winning_list)
        feasibility_winning=sum(feasibility_winning_list)/len(feasibility_winning_list)
        viability_winning = sum(viability_winning_list) / len(viability_winning_list)
        overall_winning = sum(overall_winning_list) / len(overall_winning_list)

        final_rbo = sum(rbo_list) / len(rbo_list)

        with open('../output/evaluation_file_rbo.csv', 'a') as f:
            f.write(str(sup_rate) + '_sum_all_rbo %s' % final_rbo + '\n')


        with open('../output/evaluation_file_g2.csv', 'a') as f:
            f.write(str(sup_rate) + '_finalmultiply_all_winning %s' % multiply_all_winning + '\n')
            f.write(str(sup_rate) + '_finalsum_all_winning %s' % sum_all_winning + '\n')
            f.write(str(sup_rate) + '_finalsum_3_winning %s' % sum_3_winning + '\n')
            f.write(str(sup_rate) + '_finaldesirability_winning %s' % desirability_winning + '\n')
            f.write(str(sup_rate) + '_finalfeasibility_winning %s' % feasibility_winning + '\n')
            f.write(str(sup_rate) + '_finalviability_winning %s' % viability_winning + '\n')
            f.write(str(sup_rate) + '_finaloverall_winning %s' % overall_winning + '\n')

        with open('../output/evaluation_file_combined.csv', 'a') as f:
           f.write(str(sup_rate) + '_desirability_before_MSEVAL %s' %  MSE_before_desirabilitylist+ '\n')
           f.write(str(sup_rate) + '_desirability_before_RMSEVAL %s' % RMSE_before_desirabilitylist + '\n')
           f.write(str(sup_rate) + '_desirability_before_MSETEST %s' % MSE_test_before_desirabilitylist + '\n')
           f.write(str(sup_rate) + '_desirability_before_RMSETEST %s' % RMSE_test_before_desirabilitylist + '\n')
           f.write(str(sup_rate) + '_desirability_MSEVAL %s' % MSE_after_desirabilitylist + '\n')
           f.write(str(sup_rate) + '_desirability_RMSEVAL %s' %  RMSE_after_desirabilitylist+ '\n')
           f.write(str(sup_rate) + '_desirability_MSETEST %s' % MSE_test_after_desirabilitylist + '\n')
           f.write(str(sup_rate) + '_desirability_RMSETEST %s' % RMSE_test_after_desirabilitylist + '\n')

           f.write(str(sup_rate) + '_feasibility_before_MSEVAL %s' % MSE_before_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_before_RMSEVAL %s' % RMSE_before_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_before_MSETEST %s' % MSE_test_before_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_before_RMSETEST %s' % RMSE_test_before_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_MSEVAL %s' % MSE_after_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_RMSEVAL %s' % RMSE_after_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_MSETEST %s' % MSE_test_after_feasibilitylist + '\n')
           f.write(str(sup_rate) + '_feasibility_RMSETEST %s' % RMSE_test_after_feasibilitylist + '\n')

        with open('../output/evaluation_file_combined.csv', 'a') as f:
          f.write(str(sup_rate) + '_viability_before_MSEVAL %s' % MSE_before_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_before_RMSEVAL %s' % RMSE_before_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_before_MSETEST %s' % MSE_test_before_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_before_RMSETEST %s' % RMSE_test_before_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_MSEVAL %s' % MSE_after_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_RMSEVAL %s' % RMSE_after_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_MSETEST %s' % MSE_test_after_viabilitylist + '\n')
          f.write(str(sup_rate) + '_viability_RMSETEST %s' % RMSE_test_after_viabilitylist + '\n')

        with open('../output/evaluation_file_combined.csv', 'a') as f:
              f.write(str(sup_rate) + '_overall_before_MSEVAL %s' % MSE_before_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_before_RMSEVAL %s' % RMSE_before_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_before_MSETEST %s' % MSE_test_before_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_before_RMSETEST %s' % RMSE_test_before_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_MSEVAL %s' % MSE_after_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_RMSEVAL %s' % RMSE_after_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_MSETEST %s' % MSE_test_after_overalllist + '\n')
              f.write(str(sup_rate) + '_overall_RMSETEST %s' % RMSE_test_after_overalllist + '\n')


