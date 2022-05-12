python3 ../src/variational_inference_idea_multi_multi_label.py \
		--inputfile_ideas  '../input/labeled_data_idea.csv'\
		--answer_matrix '../input/answer_matrix_overall.csv'\
		--answer_matrix_viability '../input/answer_matrix_viability.csv'\
		--answer_matrix_feasibility '../input/answer_matrix_feasibility.csv'\
		--answer_matrix_desirability '../input/answer_matrix_desirability.csv'\
		--answer_matrix_unlabeled '../input/answer_matrix_unlabeled.csv'\
		--answer_matrix_unlabeled_viability '../input/answer_matrix_unlabeled_viability.csv'\
		--answer_matrix_unlabeled_feasibility '../input/answer_matrix_unlabeled_feasibility.csv'\
		--answer_matrix_unlabeled_desirability '../input/answer_matrix_unlabeled_desirability.csv'\
		--sup_rate 6\
		--iterr 10\
		--max_budget 5\
		--classifier 'logreg'\
		--evaluation_file '../output/small_example.csv'\
		--evaluation_file_viability '../output/evaluation_file_viability.csv'\
		--evaluation_file_feasibility '../output/evaluation_file_feasibility.csv'\
		--evaluation_file_desirability '../output/evaluation_file_desirability.csv'\
		--unlabeled_ideas '../input/unlabeled_data_idea.csv'\

