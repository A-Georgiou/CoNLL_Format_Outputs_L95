from spacy_conll import init_parser
from stanza.utils.conll import CoNLL
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialise English parser, already including the ConllFormatter as a pipeline component.
# Indicate that we want to get the CoNLL headers in the string output.
# `use_gpu` and `verbose` are specific to stanza (and stanfordnlp). These keywords arguments
# are passed onto their Pipeline() initialisation
nlp = init_parser("en","stanza",
                  parser_opts={"use_gpu": True, "verbose": False},
                  include_headers=True)

sentences = [
    "JetBlue cancelled our flight this morning which was already late.",    
    "The old car broke down in the car park.",
    "The horses were broken in and ridden in two weeks.",
    "At least two men broke in and stole my TV.",
    "Kim and Sandy both broke up with their partners.",
    "The veterans who I thought that we would meet at the reunion were dead.",
    "English also has many words of more or less unique function, including interjections (oh, ah), negatives (no, not), politeness markers (please, thank you), and the existential ‘there’ (there are horses but not unicorns) among others.",
    "The Penn Treebank tagset was culled from the original 87-tag tagset for the Brown Corpus. For example the original Brown and C5 tagsets include a separate tag for each of the different forms of the verbs do (e.g. C5 tag VDD for did and VDG tag for doing), be and have.",
    "The slightly simplified version of the Viterbi algorithm that we present takes as input a single HMM and a sequence of observed words O = (o1, o2, ...oT ) and returns the most probable state/tag sequence Q = (q1, q2, qT ) together with its probability.",
    "Thus the EM-trained “pure HMM” tagger is probably best suited to cases where no training data is available, for example, when tagging languages for which no data was previously hand-tagged.",
    "Coming home from very lonely places, all of us go a little mad: whether from great personal success, or just an all-night drive, we are the sole survivors of a world no one else has ever seen."]


gold_doc = CoNLL.conll2doc("gold.conllu")
malt_doc = CoNLL.conll2doc("output_new.conll")

sentence_length_error = []
sentence_depen_error = []
sentence_length_error_malt = []
sentence_depen_error_malt = []

dep_gold_dict = defaultdict(lambda: 0)

deprel_dict_malt = defaultdict(lambda: 0)
correct_dep_malt = defaultdict(lambda: 0)
error_dep_malt = defaultdict(lambda: 0)

deprel_dict_stan = defaultdict(lambda: 0)
correct_dep_stan = defaultdict(lambda: 0)
error_dep_stan = defaultdict(lambda: 0)

distance_root_errors_stan = defaultdict(lambda: 0)
distance_root_errors_malt = defaultdict(lambda: 0)

distance_root_correct_stan = defaultdict(lambda: 0)
distance_root_correct_malt = defaultdict(lambda: 0)

count_malt = 0
count_stan = 0

for i, sentence in enumerate(malt_doc.sentences):
    gold_sentence = gold_doc.sentences[i]
    gold_root = 0
    gold_dict = gold_sentence.to_dict()
    curr_dict = sentence.to_dict()
    current_sentence = []
    root_dist = []
    
    for token in curr_dict:
        if(token['head'] == 0):
            gold_root = token['id']
        current_sentence.append(token['text'])
        
    malt_heads = []
    gold_heads = []
    malt_dep = []
    gold_dep = []
    
    for token in curr_dict:
        deprel_dict_malt[token['upos'].lower()] += 1
        malt_dep.append(token['deprel'])
        root_dist.append(abs(gold_root - token['id']))
        gold_heads.append(str(curr_dict[token['head']]['text']).lower())
    
    for token in gold_dict:
        dep_gold_dict[token['deprel'].lower()] += 1
        gold_dep.append(token['deprel'])
        malt_heads.append(str(gold_dict[token['head']]['text']).lower())
    
    for gold, malt in zip(gold_dict, curr_dict):
        if(gold['deprel'].lower() == malt['deprel'].lower()):
            correct_dep_malt[malt['deprel'].lower()] += 1
        else:
            error_dep_malt[malt['deprel'].lower()] += 1

    count_len_errors = 0
    count_depen_errors = 0
    
    for malt_head, gold_head, malt_dep_head, gold_dep_head, dist in zip(malt_heads, gold_heads, malt_dep, gold_dep, root_dist):
        count_malt += 1
        if(malt_head != gold_head):
            distance_root_errors_malt[dist] += 1
            count_len_errors += 1
            if(malt_dep_head != gold_dep_head):
                count_depen_errors += 1
        else:
            distance_root_correct_malt[dist] += 1
    
    sentence_depen_error_malt.append([len(current_sentence), count_depen_errors])
    sentence_length_error_malt.append([len(current_sentence), count_len_errors])


for i, sentence in enumerate(gold_doc.sentences):
    gold_root = 0
    curr_dict = sentence.to_dict()
    current_sentence = []
    root_dist = []
    
    for token in curr_dict:
        current_sentence.append(token['text'])
    
    curr_sent = ' '.join(current_sentence)
    curr_stanza = nlp(curr_sent)
    
    stanza_heads = []
    gold_heads = []
    
    stanza_dep_heads = []
    gold_dep_heads = []
    
    for token in curr_stanza:
        if(token.head.i == token.i):
            gold_root = token.i
    
    for token in curr_stanza:
        root_dist.append(abs(gold_root-token.i))
        stanza_dep_heads.append(token.dep_)
        stanza_heads.append(str(token.head).lower())
        deprel_dict_stan[token.pos_.lower()] += 1
        
    for token in curr_dict:
        gold_dep_heads.append(token['deprel'])
        gold_heads.append(str(curr_dict[token['head']-1]['text']).lower())
    

    count_len_errors = 0
    count_depen_errors = 0
    tp_len = 0
    
    for gold, stan in zip(gold_dep_heads, stanza_dep_heads):
        if(gold.lower() == stan.lower()):
            correct_dep_stan[stan.lower()] += 1
        else:
            error_dep_stan[stan.lower()] += 1
    
    for stan_head, gold_head, stanza_dep_head, gold_dep_head, dist in zip(stanza_heads, gold_heads, stanza_dep_heads, gold_dep_heads, root_dist):
        count_stan += 1
        if(stan_head != gold_head):
            if(stanza_dep_head != gold_dep_head):
                count_depen_errors += 1
            distance_root_errors_stan[dist] += 1
            count_len_errors += 1
        else:
            distance_root_correct_stan[dist] += 1

    sentence_depen_error.append([len(current_sentence), count_depen_errors])
    sentence_length_error.append([len(current_sentence), count_len_errors])
    


sentence_length_error = sorted(sentence_length_error, key=lambda x: x[0])
sentence_depen_error = sorted(sentence_depen_error, key=lambda x: x[0])

sentence_length_error_malt = sorted(sentence_length_error_malt, key=lambda x: x[0])
sentence_depen_error_malt = sorted(sentence_depen_error_malt, key=lambda x: x[0])


las_malt = sum(x[1] for x in sentence_length_error_malt)/count_malt
las_stan = sum(x[1] for x in sentence_length_error)/count_stan

uas_malt = sum(x[1] for x in sentence_depen_error_malt)/count_malt
uas_stan = sum(x[1] for x in sentence_depen_error)/count_stan

print(las_malt, uas_malt)
print(las_stan, uas_stan)


errors_accuracy = [1.0-(x[1]/x[0]) for x in sentence_length_error]
errors_depen_accuracy = [1.0-(x[1]/x[0]) for x in sentence_depen_error]

errors_accuracy_malt = [1.0-(x[1]/x[0]) for x in sentence_length_error_malt]
errors_depen_accuracy_malt = [1.0-(x[1]/x[0]) for x in sentence_depen_error_malt]

fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_length_error], [x[1] for x in sentence_length_error], color='black', label='Stanford Parser')
plt.plot([x[0] for x in sentence_length_error_malt], [x[1] for x in sentence_length_error_malt], color='darkred', label='MaltParser')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Head Errors')
plt.legend()
fig.savefig('SentenceErrorGraphTokens.jpg')


fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_depen_error], [x[1] for x in sentence_depen_error], color='black', label='Stanford Parser')
plt.plot([x[0] for x in sentence_depen_error_malt], [x[1] for x in sentence_depen_error_malt], color='darkred', label='MaltParser')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Label Errors')
plt.legend()
fig.savefig('SentenceErrorGraphDepens.jpg')

fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_depen_error], errors_accuracy, color='black')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Head Accuracy')

fig.savefig('SentenceErrorGraphDepens.jpg')

fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_depen_error], errors_depen_accuracy, color='black')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Label Accuracy')
fig.savefig('SentenceErrorGraphDepens.jpg')

fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_depen_error_malt], errors_accuracy_malt, color='darkred')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Head Accuracy')

fig.savefig('SentenceErrorGraphDepens.jpg')

fig = plt.figure()
fig.set_size_inches(8, 3)
plt.plot([x[0] for x in sentence_depen_error_malt], errors_depen_accuracy_malt, color='darkred')
plt.xlabel('Sentence Token Length')
plt.ylabel('Dependency Label Accuracy')
fig.savefig('SentenceErrorGraphDepens.jpg')


X = ['SYM','NUM','SCONJ','INTJ', 'AUX', 'PROPN', 'CCONJ', 'PRON', 'ADV', 'ADJ', 'ADP', 'DET', 'VERB', 'PUNCT', 'NOUN']

y_malt = [deprel_dict_malt[y.lower()] for y in X]
y_stan = [deprel_dict_stan[y.lower()] for y in X]

X_axis = np.arange(len(X))

plt.figure(figsize=(20, 3))  # width:20, height:3
plt.bar(X_axis - 0.2, y_malt, width=0.4, label = 'MaltParser', align='edge', color='darkred')
plt.bar(X_axis + 0.2, y_stan, width=0.4, label = 'Stanford Parser', align='edge', color='black')
plt.xticks(X_axis, X, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Part of Speech Tag", fontsize=15)
plt.ylabel("Tag Occurence", fontsize=15)
plt.legend(fontsize=20)
plt.show()


precision_malt, recall_malt, f1_malt = [],[],[]
precision_stan, recall_stan, f1_stan = [],[],[]
tags = []

for key, val in dep_gold_dict.items():
    
    tp_malt = correct_dep_malt[key]
    tp_stan = correct_dep_stan[key]
    
    actual_results = val
    pred_malt = error_dep_malt[key] + tp_malt
    pred_stan = error_dep_stan[key] + tp_stan
    
    prec_malt = tp_malt/actual_results
    prec_stan = tp_stan/actual_results
    
    rec_malt = 0
    rec_stan = 0
    f1_malt_s = 0
    f1_stan_s = 0
    
    if(pred_malt != 0):    
        rec_malt = tp_malt/pred_malt
        
    if(prec_malt != 0 or rec_malt != 0):
        f1_malt_s = (2*(prec_malt*rec_malt))/(prec_malt+rec_malt)
    
    if(pred_stan != 0):
        rec_stan = tp_stan/pred_stan
    
    if(prec_stan != 0 or rec_stan != 0):
        f1_stan_s = (2*(prec_stan*rec_stan))/(prec_stan+rec_stan)
    
    print(key, "&", "%.2f" % prec_malt, "&", "%.2f" % rec_malt, "&", "%.2f" % f1_malt_s, "\\\\")
    #print(key, "&", "%.2f" % prec_stan, "&", "%.2f" % rec_stan, "&", "%.2f" % f1_stan_s, "\\\\")
    
    tags.append(key)
    precision_malt.append(prec_malt)
    precision_stan.append(prec_stan)
    recall_malt.append(rec_malt)
    recall_stan.append(rec_stan)
    
    f1_malt.append(f1_malt_s)
    f1_stan.append(f1_stan_s)
    
malt_tp = 0
overall_malt = 0

stan_tp = 0
overall_stan = 0

overall_gold = 0

for key, val in dep_gold_dict.items():
    overall_gold += val

for key, val in correct_dep_malt.items():
    malt_tp += val
    overall_malt += val
    
for key, val in correct_dep_stan.items():
    stan_tp += val
    overall_stan += val

for key, val in error_dep_malt.items():
    overall_malt += val
    
for key, val in error_dep_stan.items():
    overall_stan += val
    
malt_micro_average_prec = malt_tp/overall_malt
stan_micro_average_prec = stan_tp/overall_stan

malt_micro_average_rec = malt_tp/overall_gold
stan_micro_average_rec = stan_tp/overall_gold

malt_macro_average_prec = sum(precision_malt)/len(precision_malt)
stan_macro_average_prec = sum(precision_stan)/len(precision_stan)

malt_macro_average_rec = sum(recall_malt)/len(recall_malt)
stan_macro_average_rec = sum(recall_stan)/len(recall_stan)

malt_micro_f1 = (2*(malt_micro_average_prec * malt_micro_average_rec)/(malt_micro_average_prec + malt_micro_average_rec))
stan_micro_f1 = (2*(stan_micro_average_prec * stan_micro_average_rec)/(stan_micro_average_prec + stan_micro_average_rec))

malt_macro_f1 = (2*(malt_macro_average_prec * malt_macro_average_rec)/(malt_macro_average_prec + malt_macro_average_rec))
stan_macro_f1 = (2*(stan_macro_average_prec * stan_macro_average_rec)/(stan_macro_average_prec + stan_macro_average_rec))


distance_sum_malt = defaultdict(lambda: 0)
distance_sum_stan = defaultdict(lambda: 0)

for key, item in distance_root_correct_malt.items():
    distance_sum_malt[key] += item
    
for key, item in distance_root_errors_malt.items():
    distance_sum_malt[key] += item
    
for key, item in distance_root_correct_stan.items():
    distance_sum_stan[key] += item
    
for key, item in distance_root_errors_stan.items():
    distance_sum_stan[key] += item

distance_malt = sorted(list(distance_sum_malt.items()), key=lambda x: x[0])
distance_stan = sorted(list(distance_sum_stan.items()), key=lambda x: x[0])

accuracy_malt = []
accuracy_stan = []
for dist, total in distance_malt:
    acc = distance_root_correct_malt[dist]/total
    accuracy_malt.append([dist, acc])

for dist, total in distance_stan:
    acc = distance_root_correct_stan[dist]/total
    accuracy_stan.append([dist,acc])

fig = plt.figure()
fig.set_size_inches(8, 3)

max_y_malt = max(distance_malt, key=lambda x: x[1])[1]
max_y_stan = max(distance_stan, key=lambda x: x[1])[1]

plt.plot([x[0] for x in accuracy_malt], [x[1] for x in accuracy_malt], color='darkred', label='MaltParser')
plt.plot([x[0] for x in accuracy_stan], [x[1] for x in accuracy_stan], color='black', label='Stanford Parser')
plt.xlabel('Distance From Gold Standard Root')
plt.ylabel('Dependency Head Accuracy')
plt.legend()
fig.savefig('SentenceErrorGraphDepens.jpg')
