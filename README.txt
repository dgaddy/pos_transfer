hmm.py expects data in pos_data folder
load_and_save.py has stuff to help load conll06/07, multext east, and nltk corpora into the expected

required args:
target - the target language
out_file - where the resulting pos predictions are written

example use:
python -u hmm.py sources:conll-spanish target:conll-english out_file:result1 &> out1 &
