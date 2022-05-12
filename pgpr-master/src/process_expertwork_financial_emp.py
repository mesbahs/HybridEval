import pandas as pd
import csv
import re
inputToken = '((('
df1=pd.read_csv('/Users/smesbah/Downloads/pgpr-master/input/Batch_financialempowerement_expert.csv')
df= pd.read_csv('/Users/smesbah/Downloads/pgpr-master/input/Batch_fanancialempowerment_crowd.csv')
df=df.groupby("HITId").agg({"HITId":"first","Answer.Desirable":"first", "Answer.feasible":"first", "Answer.overall":"first","Answer.viability":"first","Input.variable_name":"first"})
list_desirability=[]
list_feasibility=[]
list_viability=[]
list_overall=[]
list_desirability_un=[]
list_feasibility_un=[]
list_viability_un=[]
list_overall_un=[]
list_unlabeled=[]
# df1['Input.variable_name'] = df1['Input.variable_name'].str.replace(re.escape(inputToken), '>>')
# df['Input.variable_name'] = df['Input.variable_name'].str.replace(re.escape(inputToken), '>>')
# df['Input.variable_name'] = df['Input.variable_name'].str.replace('\\', '')
# df['Input.variable_name'] = df['Input.variable_name'].str.replace('>', '')
# df['Input.variable_name'] = df['Input.variable_name'].str.replace('<', '')
count=0
mylist=[]
with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/labeled_data_idea.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Id', 'title', 'idea', 'rating', 'labels', 'labels_viability', 'labels_feasibility', 'labels_desirability'])
    for index, row in df.iterrows():
        # df['HITId'], df['WorkerId'], df['Answer.Desirable'], df['Answer.feasible'], df['Answer.overall'], df[
        #     'Answer.viability']
        # temp=row['Input.variable_name']
        # temp=df1[df1['Input.variable_name'].str.contains(temp)]
        # print(temp)\
        with open('/Users/smesbah/Downloads/pgpr-master/input/Batch_financialempowerement_expert.csv', 'r') as file:
            reader = csv.reader(file)
            check=False
            for row2 in reader:
                # print(row['Input.variable_name'])
                text=row2[27].replace('\\', '')
                text=text.replace('>', '')
                text = text.replace('<', '')
               # temp="<h4>Tell us about your work experience:</h4><p>This idea sprang from an interdisciplinary group of experts. Our group had a designer/facilitator- prototyper- social worker and Program & Education Specialist for Alzheimer's Association- healthcare technology industry veteran with over 25 years of management experience- and a music therapist.</p><h4>What skills- input- or guidance from the OpenIDEO community would be most helpful in building out or refining your idea?</h4><p>Feedback and a team to move this idea forward.</p><h4>What early- lightweight experiment might you try out in your own community to find out if the idea will meet your expectations?</h4><p>A clickable prototype for the app and some user testing with the community.</p><h4>Who is your idea designed for and how does it better support family caregivers as they care for a loved one with dementia?</h4><p>Our audience are full-time- part-time- remote family caregivers who need to be encouraged to take time to focus on self-care.</p><h4>How long has your idea existed?</h4><p>0-3 months</p><h4>This idea emerged from</h4><p>A group brainstorm An OpenIDEO Outpost or Chapter</p>"
               # temp2="<h2>Treat'r: Navigating your way to self-care</h2><p>An app that promotes pro-active and affordable self-care that rewards caregiver’s hard work through personal reminders and activities.</p><h4>Who is your idea designed for and how does it better support family caregivers as they care for a loved one with dementia?</h4><p>Our audience are full-time- part-time- remote family caregivers who need to be encouraged to take time to focus on self-care.</p><h4>What early- lightweight experiment might you try out in your own community to find out if the idea will meet your expectations?</h4><p>A clickable prototype for the app and some user testing with the community.</p><h4>What skills- input- or guidance from the OpenIDEO community would be most helpful in building out or refining your idea?</h4><p>Feedback and a team to move this idea forward.</p><h4>How long has your idea existed?</h4><p>0-3 months</p><h4>This idea emerged from</h4><p>A group brainstorm An OpenIDEO Outpost or Chapter</p><h4>Tell us about your work experience:</h4><p>This idea sprang from an interdisciplinary group of experts. Our group had a designer/facilitator- prototyper- social worker and Program & Education Specialist for Alzheimer's Association- healthcare technology industry veteran with over 25 years of management experience- and a music therapist.</p>"
               #  print(row['Input.variable_name'])
                temp=re.findall(r'p>(.*?)</p', row['Input.variable_name'])
                if temp!=None and temp!=[]:
                    try:

                        if temp[0] in row2[27] and temp[1] in row2[27]:
                            check=True
                            count = count + 1
                            #print(row2[27])
                            #print(row2[27])
                            mylist.append(row['HITId'])
                            print(row['HITId'],row2[0])
                            my_text=''
                            for tt in temp:
                                my_text=my_text+' '+ tt


                            # writer.writerow(
                            #     [row['HITId'], row['HITId'], row['Input.variable_name'], row2[33], row2[33],
                            #      row2[34], row2[32], row2[28]])
                            writer.writerow(
                                [row['HITId'], row['HITId'], my_text, row2[33], row2[33],
                                 row2[34], row2[32], row2[28]])

                    except:
                        pass
        if check==False:
            list_unlabeled.append([row['HITId'],row['HITId'],row['Input.variable_name'],row['Answer.overall'],row['Answer.overall'],row['Answer.viability'],row['Answer.feasible'],row['Answer.Desirable']])

print(count)
print(len(list(set(mylist))))
df= pd.read_csv('/Users/smesbah/Downloads/pgpr-master/input/Batch_fanancialempowerment_crowd.csv')
for index, row in df.iterrows():
    if row['HITId'] in mylist:

        list_desirability.append([row['HITId'],row['WorkerId'],row['Answer.Desirable']])
        list_viability.append([row['HITId'], row['WorkerId'], row['Answer.viability']])
        list_feasibility.append([row['HITId'], row['WorkerId'], row['Answer.feasible']])
        list_overall.append([row['HITId'], row['WorkerId'], row['Answer.overall']])
    else:
        list_desirability_un.append([row['HITId'], row['WorkerId'], row['Answer.Desirable']])
        list_viability_un.append([row['HITId'], row['WorkerId'], row['Answer.viability']])
        list_feasibility_un.append([row['HITId'], row['WorkerId'], row['Answer.feasible']])
        list_overall_un.append([row['HITId'], row['WorkerId'], row['Answer.overall']])

        # print(row)




print(len(list_overall),len(list_overall_un))
with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_overall.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['idea','worker','rating'])
        for row in list_overall:
                #print(row)

                writer.writerow(row)

with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_desirability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_desirability:
        writer.writerow(row)


with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_feasibility.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_feasibility:
        writer.writerow(row)
with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_viability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_viability:
        writer.writerow(row)

with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_unlabeled_overall.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['idea', 'worker', 'rating'])
        for row in list_overall_un:
            writer.writerow(row)

with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_unlabeled_desirability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_desirability_un:
        writer.writerow(row)

with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_unlabeled_feasibility.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_feasibility_un:
        writer.writerow(row)
with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/answer_matrix_unlabeled_viability.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['idea', 'worker', 'rating'])
    for row in list_viability_un:
        writer.writerow(row)



with open('/Users/smesbah/Downloads/pgpr-master/input_fanancial_emp/unlabeled_data_idea.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'title', 'idea', 'rating', 'labels', 'labels_viability', 'labels_feasibility',
                         'labels_desirability'])
        for row in list_unlabeled:
            writer.writerow(row)

#
# idx_1 = df.groupby(['HITId'])['Answer.Desirable'].transform(max) == df['Answer.Desirable']
# idx_1 =df.groupby("HITId").agg({"HITId":"first","Answer.Desirable":"first", "Answer.feasible":"first", "Answer.overall":"first","Answer.viability":"first","Input.variable_name":"first"})
# # idx_2 = df.groupby(['HITId'])['Answer.feasible'].transform(max) == df['Answer.feasible']
# # idx_3 = df.groupby(['HITId'])['Answer.overall'].transform(max) == df['Answer.overall']
# # idx_4 = df.groupby(['HITId'])['Answer.viability'].transform(max) == df['Answer.viability']
# # print(idx_1.head())
#
#
#
#
# with open('/Users/smesbah/Downloads/pgpr-master/input_crowd/labeled_data_idea.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Id','title','idea','rating','labels','labels_viability','labels_feasibility','labels_desirability'])
#         for index, row in idx_1.iterrows():
#                 # print(row)
#
#                 writer.writerow([row['HITId'],row['HITId'],row['Input.variable_name'],row['Answer.overall'],row['Answer.overall'],row['Answer.viability'],row['Answer.feasible'],row['Answer.Desirable']])