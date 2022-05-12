import pandas as pd
import csv

df= pd.read_csv('/Users/smesbah/Downloads/pgpr-master/input/Batch_4176044_batch_results.csv')
for index, row in df.iterrows():

    df['HITId'],df['WorkerId'],df['Answer.Desirable'], df['Answer.feasible'],df['Answer.overall'],df['Answer.viability']



with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/answer_matrix_overall.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['idea','worker','rating'])
        for index, row in df.iterrows():

                writer.writerow([row['HITId'],row['WorkerId'],row['Answer.overall']])

with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/answer_matrix_desirability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for index, row in df.iterrows():
        writer.writerow([row['HITId'], row['WorkerId'], row['Answer.Desirable']])


with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/answer_matrix_feasibility.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for index, row in df.iterrows():
        writer.writerow([row['HITId'], row['WorkerId'], row['Answer.feasible']])
with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/answer_matrix_viability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for index, row in df.iterrows():
        writer.writerow([row['HITId'], row['WorkerId'], row['Answer.viability']])


idx_1 = df.groupby(['HITId'])['Answer.Desirable'].transform(max) == df['Answer.Desirable']
idx_1 =df.groupby("HITId").agg({"HITId":"first","Answer.Desirable":"first", "Answer.feasible":"first", "Answer.overall":"first","Answer.viability":"first","Input.variable_name":"first"})
# idx_2 = df.groupby(['HITId'])['Answer.feasible'].transform(max) == df['Answer.feasible']
# idx_3 = df.groupby(['HITId'])['Answer.overall'].transform(max) == df['Answer.overall']
# idx_4 = df.groupby(['HITId'])['Answer.viability'].transform(max) == df['Answer.viability']
# print(idx_1.head())




with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/labeled_data_idea.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id','title','idea','rating','labels','labels_viability','labels_feasibility','labels_desirability'])
        for index, row in idx_1.iterrows():
                # print(row)

                writer.writerow([row['HITId'],row['HITId'],row['Input.variable_name'],row['Answer.overall'],row['Answer.overall'],row['Answer.viability'],row['Answer.feasible'],row['Answer.Desirable']])