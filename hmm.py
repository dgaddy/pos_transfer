from collections import Counter
import numpy
import scipy.optimize
import math
import sys
import scipy.stats
import load_and_save
import prototypes_helper
import word_vect_loader

from joblib import Parallel, delayed
import multiprocessing

universal_pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', 'START']
universal_pos_closed = numpy.array([False, False, True, False, False, True, True, True, False, True, False, True, True], dtype=bool)

def fill_forward_lattice(sentence_int, trans_probs, emission_probs, start_token):
    n = trans_probs.shape[0]
    lattice = numpy.zeros((len(sentence_int), n))
    scaling_factors = numpy.zeros((len(sentence_int)))

    def normalize(arr):
        s = arr.sum()
        arr /= s
        return s

    for i, w in enumerate(sentence_int):
        if i == 0:
            lattice[i, :] = trans_probs[start_token, :] * emission_probs[:, w]
            lattice[i, start_token] = 0
        else:
            lattice[i, :] = lattice[i-1, :][numpy.newaxis, :].dot(trans_probs).ravel() * emission_probs[:, w]
            lattice[i, start_token] = 0

        scaling_factors[i] = 1.0 / normalize(lattice[i, :])

    return lattice, scaling_factors


def add_counts_forward_backward(sentence_int, trans_probs, emission_probs, start_token, trans_counts, em_counts):
    n_pos = trans_probs.shape[0]
    len_sent = len(sentence_int)
    forward_lattice, scaling_factors = fill_forward_lattice(sentence_int, trans_probs, emission_probs, start_token)

    backward_lattice = numpy.zeros((len_sent, n_pos))

    backward_lattice[len_sent-1, :] = trans_probs[:, start_token] # probability of transitioning to end token
    norm = (forward_lattice[len_sent-1] * backward_lattice[len_sent-1]).sum()
    backward_lattice[len_sent-1, :] /= norm
    # without end token, would be numpy.ones((1, n_pos))
    for i in reversed(xrange(len_sent-1)):
        backward_lattice[i, :] = (trans_probs.dot((backward_lattice[i+1, :] * emission_probs[:, sentence_int[i+1]])[:, numpy.newaxis]) * scaling_factors[i+1]).ravel()
        backward_lattice[i, start_token] = 0

    probs = forward_lattice * backward_lattice

    for i in xrange(len_sent):
        # emission counts
        em_counts[:, sentence_int[i]] += probs[i, :]
        # transition counts
        if i == 0:
            trans_counts[start_token, :] += probs[i, :]
        else:
            m = forward_lattice[i-1, :][:, numpy.newaxis].dot((backward_lattice[i, :] * emission_probs[:, sentence_int[i]])[numpy.newaxis, :])
            m *= trans_probs
            m /= m.sum()
            trans_counts += m
        if i == len_sent - 1:
            trans_counts[:, start_token] += probs[i, :]


def viterbi(sentence_int, trans_probs, emission_probs, start_token):
    n = trans_probs.shape[0]
    lattice = numpy.zeros((len(sentence_int), n))
    back_ptr = numpy.zeros((len(sentence_int), n), dtype=int)

    def normalize(arr):
        s = arr.sum()
        arr /= s
        return s

    probability_factor = 0
    for i, w in enumerate(sentence_int):
        if i == 0:
            for j in xrange(n):
                if j == start_token:
                    continue

                lattice[i, j] = trans_probs[start_token, j] * emission_probs[j, w]
        else:
            for j in xrange(n):
                if j == start_token:
                    continue

                val = 0
                val_from = -1
                for k in xrange(n):
                    v = trans_probs[k, j] * lattice[i-1, k]
                    if v > val:
                        val = v
                        val_from = k
                val *= emission_probs[j, w]
                back_ptr[i, j] = val_from
                lattice[i, j] = val

        probability_factor += math.log(normalize(lattice[i, :]))

    end_probs = lattice[-1, :] * trans_probs[:, start_token]  # include the probability of going to end token
    last = end_probs.argmax()
    seq_rev = [last]
    for i in reversed(xrange(1, len(sentence_int))):
        last = back_ptr[i, last]
        seq_rev.append(last)
    seq_rev.reverse()
    return seq_rev, math.log(end_probs.max()) + probability_factor

def get_shared_model(languages, univeral_pos_tags, do_lang_specific):
    transitions = []
    tag_distributions = []
    universal_tag_maps = []
    for l, language in enumerate(languages):
        filename = 'pos_data/'+language+ ('.spec.pos' if do_lang_specific else '.pos')
        text_sentences = load_and_save.read_sentences_from_file(filename)

        sentences, words, sentences_pos, pos = load_and_save.integer_sentences(text_sentences, pos=None if do_lang_specific else univeral_pos_tags, max_words=1000)

        if do_lang_specific:
            start_token = len(pos)
            pos.append('START-') #language specific tags must have a universal and a dash

            pos_map = [0 for _ in xrange(len(pos))]
            for i, p in enumerate(pos):
                universal = p[:p.index('-')]
                pos_map[i] = universal_pos_tags.index(universal)
        else:
            start_token = universal_pos_tags.index('START')
            pos_map = range(len(pos))

        n_pos = len(pos)

        trans_counts = numpy.zeros((n_pos, n_pos))
        word_pos = numpy.zeros((n_pos, 1000))

        for sent, sent_pos in zip(sentences, sentences_pos):
            for i in xrange(len(sent)):
                w = sent[i]
                p = sent_pos[i]
                word_pos[p, w] = 1
                if i == 0:
                    trans_counts[start_token, p] += 1
                else:
                    trans_counts[sent_pos[i-1], p] += 1
            trans_counts[sent_pos[-1], start_token] += 1

        trans_counts += 1e-8
        trans_probs = trans_counts / trans_counts.sum(axis=1)[:, numpy.newaxis]
        transitions.append(trans_probs)

        tag_counts = word_pos.sum(axis=1)
        tag_probs = tag_counts / tag_counts.sum()
        tag_distributions.append(tag_probs)

        universal_tag_maps.append(pos_map)

    return transitions, tag_distributions, universal_tag_maps

class FeatureMap:
    def __init__(self):
        self.feats = []
        self.map = {}

    def get_id(self, feature):
        if feature in self.map:
            return self.map[feature]
        else:
            id = len(self.feats)
            self.map[feature] = id
            self.feats.append(feature)
            return id

def make_emission_features(pos, words, prototypes):
    n_pos = len(pos)
    n_words = len(words)
    fm = FeatureMap()
    feature_matrix = numpy.empty((n_pos, n_words), dtype=object)
    for i in xrange(n_pos):
        for j in xrange(n_words):
            word = words[j]
            feats = ['emit %s %s' % (pos[i], word), 'emit_suf %s %s' % (pos[i], word[-3:])]
            if prototypes is not None:
                for p in prototypes[j]:
                    feats.append('prototype %s %s' % (pos[i], p))
            feature_matrix[i, j] = [fm.get_id(f) for f in feats]

    return feature_matrix, fm

def probs_from_weights(weights, mask=None):
    # TODO: do we want to offset weights before exp for numerical reasons?

    exp_weights = numpy.exp(weights)
    if mask is not None:
        exp_weights *= mask
    return exp_weights / exp_weights.sum(axis=1)[:, numpy.newaxis]

def optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, source_trans_probs,
                      update_transitions, direct_gradient, emission_features, n_em_feats,
                      l2_weight_regularization, emission_mask):

    # for transition, there is just one feature per transition, so we can use the normalized exp of the weights
    def loss_trans(weights):
        weights = weights.reshape(trans_counts.shape)  # because optimize flattens them
        p = probs_from_weights(weights)
        log_p = numpy.log(p)
        result = -((trans_counts * log_p).sum())

        # TODO regularization

        return result

    def d_loss_trans(weights):
        weights = weights.reshape(trans_counts.shape)
        p = probs_from_weights(weights)
        p1_counts = trans_counts.sum(axis=1)
        result = (p * p1_counts[:, numpy.newaxis]) - trans_counts

        # TODO regularization

        return result.ravel()

    # for the emission, weights are for features which are used to create the emission_weight_matrix
    # the emission_weight_matrix can be exponentiated and normalized to get the distribution
    def em_matrix_from_weights(weights):
        matrix = numpy.zeros(em_counts.shape)
        for i in xrange(emission_features.shape[0]):
            for j in xrange(emission_features.shape[1]):
                matrix[i, j] = sum(weights[f] for f in emission_features[i, j])
        return matrix

    def loss_em(weights):
        emission_weight_matrix = em_matrix_from_weights(weights)
        p = probs_from_weights(emission_weight_matrix)
        log_p = numpy.log(p)
        result = -((em_counts * log_p).sum())

        result += (weights * weights).sum() * l2_weight_regularization

        return result

    def d_loss_em(weights):
        emission_weight_matrix = em_matrix_from_weights(weights)
        p = probs_from_weights(emission_weight_matrix)
        pos_counts = em_counts.sum(axis=1)
        dL_dmatrix = (p * pos_counts[:, numpy.newaxis]) - em_counts

        # regularization
        result = 2 * l2_weight_regularization * weights

        # add feature derivatives based on dL_dmatrix
        for i in xrange(emission_features.shape[0]):
            for j in xrange(emission_features.shape[1]):
                dl_dm = dL_dmatrix[i, j]
                for f in emission_features[i, j]:
                    result[f] += dl_dm

        return result.ravel()

    if update_transitions:
        if prev_trans_weights is None:
            prev_trans_weights = numpy.zeros(trans_counts.shape)
        result = scipy.optimize.minimize(loss_trans, prev_trans_weights, method='L-BFGS-B', jac=d_loss_trans,
                                         options=({'maxiter':1} if direct_gradient else {}))
        trans_weights = result.x.reshape(trans_counts.shape)
        trans_log_prob = result.fun

        trans_probs = probs_from_weights(trans_weights)
    else:
        trans_probs = source_trans_probs
        trans_weights = None

        log_p = numpy.log(trans_probs)
        trans_log_prob = -((trans_counts * log_p).sum())

    if prev_em_weights is None:
        prev_em_weights = numpy.zeros(n_em_feats)
    result = scipy.optimize.minimize(loss_em, prev_em_weights, method='L-BFGS-B', jac=d_loss_em,
                                     options=({'maxiter':1} if direct_gradient else {}))
    emission_weights = result.x
    em_log_prob = result.fun

    emission_probs = probs_from_weights(em_matrix_from_weights(emission_weights), emission_mask)

    description_string =  'log probability %s (%s %s)' % (em_log_prob + trans_log_prob, em_log_prob, trans_log_prob)

    return trans_probs, emission_probs, trans_weights, emission_weights, description_string


def model_from_counts(transition_counts, emission_counts):
    transition_counts += 1e-8
    transition_probs = transition_counts / transition_counts.sum(axis=1)[:, numpy.newaxis]

    emission_counts += 1e-8
    emission_probs = emission_counts / emission_counts.sum(axis=1)[:, numpy.newaxis]

    return transition_probs, emission_probs


default_args = {'reg_coeff': 0, 'iterations': 20, 'init_noise_level': 0, 'repeat': 1, 'update_transitions': 'False',
                'l0_coeff': 0, 'direct_gradient': 'False', 'do_lang_specific': 'False',
                'prototypes': 'False', 'random_init': 'False'}

args = default_args.copy()
for arg in sys.argv[1:]:
    i = arg.index(':')
    key = arg[:i]
    val = arg[i+1:]
    args[key] = val

(iterations, initialization_noise_level, repeat, out_file, source, target, update_transitions,
 l2_weight_regulariztion_coeff,
 direct_gradient, do_language_specific_tags, use_prototypes, random_init) = \
(int(args['iterations']), float(args['init_noise_level']), int(args['repeat']),
               args['out_file'] if 'out_file' in args else None, args['sources'].split(',') if 'sources' in args else None, args['target'],
               args['update_transitions'] == 'True', float(args['reg_coeff']),
               args['direct_gradient'] == 'True', args['do_lang_specific'] == 'True', args['prototypes'] == 'True',
               args['random_init'] == 'True')

print 'iterations', iterations
print 'initalization noise', initialization_noise_level
print 'repeats', repeat
print 'output file', out_file
print 'update transitions', update_transitions
print 'l2 weight regularization', l2_weight_regulariztion_coeff
print 'direct gradient', direct_gradient
print 'language specific tags', do_language_specific_tags

# the prototypes used by Haghighi and Klein, with capitalized words and punctuation removed (we lowercase everything)
prototypes = {'%':'NOUN', 'company':'NOUN', 'year':'NOUN', 'new':'ADJ', 'other':'ADJ', 'last':'ADJ',
              'will':'VERB', 'would':'VERB', 'could':'VERB', 'are':'VERB', "'re":'VERB', "'ve":'VERB',
              "n't":'ADV', 'also':'ADV', 'not':'ADV', 'when':'ADV', 'how':'ADV', 'where':'ADV',
              'of':'ADP', 'in':'ADP', 'for':'ADP', 'c':'X', 'b':'X', 'f':'X',
              'million':'NUM', 'billion':'NUM', 'two':'NUM', 'to':'PRT', 'na':'PRT',
              'been':'VERB', 'based':'VERB', 'compared':'VERB', 'earlier':'ADV', 'duller':'ADV',
              'is':'VERB', 'has':'VERB', 'says':'VERB', 'least':'ADJ', 'largest':'ADJ', 'biggest':'ADJ',
              'mr.':'NOUN', 'u.s.':'NOUN', 'corp.':'NOUN', "'s":'PRT',
              'its':'PRON', 'their':'PRON', 'his':'PRON', 'quite':'DET',
              'which':'DET', 'whatever':'DET', 'there':'DET',
              'years':'NOUN', 'shares':'NOUN', 'companies':'NOUN', 'including':'VERB', 'being':'VERB', 'according':'VERB',
              'the':'DET', 'a':'DET', 'whose':'PRON', 'bono':'X', 'del':'X', 'kangi':'X',
              'up':'PRT', 'on':'PRT', 'said':'VERB', 'was':'VERB', 'had':'VERB',
              'philippines':'NOUN', 'angels':'NOUN', 'rights':'NOUN', 'be':'VERB', 'take':'VERB', 'provide':'VERB',
              'worst':'ADV', 'and':'CONJ', 'or':'CONJ', 'but':'CONJ',
              'smaller':'ADJ', 'greater':'ADJ', 'larger':'ADJ', 'who':'PRON', 'what':'PRON',
              'it':'PRON', 'he':'PRON', 'they':'PRON', 'oh':'X', 'well':'X', 'yeah':'X'}

print 'prototypes', use_prototypes, prototypes

language = target
print 'to', language
text_sentences = load_and_save.read_sentences_from_file('pos_data/'+language+'.pos')
sentences, words, sentences_pos, pos = load_and_save.integer_sentences(text_sentences, pos=universal_pos_tags, max_words=1000)
n_words = len(words)

prototype_ids = [words.index(p) for p in prototypes if p in words]
for p in prototypes:
    if p in words:
        prototype_ids.append(words.index(p))
    else:
        print 'warning: prototype not in words', p

#word_vects = prototypes_helper.get_word_context_vectors(sentences, n_words)
word_vect_map, word_vect_size = word_vect_loader.load('context_svd_output.vec')
word_vect_matrix = numpy.zeros((n_words, word_vect_size))
for w in xrange(n_words):
    if words[w] in word_vect_map:
        v = word_vect_map[words[w]]
        word_vect_matrix[w, :] = v / math.sqrt((v*v).sum())
prototypes_for_words = prototypes_helper.get_prototypes(words, word_vect_matrix, prototype_ids)

emission_feature_matrix, emission_feature_map = make_emission_features(pos, words, prototypes_for_words if use_prototypes else None)

WALS_map = load_and_save.load_WALS_map('pos_data/WALS_map')
annotated_languages = source if source is not None else [l for l in WALS_map if l != target]
if len(annotated_languages) == 1 and annotated_languages[0] == '':
    annotated_languages = []
print 'from', annotated_languages
num_shared_langs = len(annotated_languages)

other_langs_trans_dists, other_langs_tag_dists, universal_pos_maps = get_shared_model(annotated_languages, universal_pos_tags, do_language_specific_tags)

print "finished loading source models"

def test(name, trans_probs, em_probs, start_token, universal_pos_map):
    correct = 0
    total = 0
    prototype_correct = 0
    prototype_total = 0
    iteration_result = []
    viterbi_log_prob = 0
    word_pos_counts = numpy.zeros((len(words), len(universal_pos_tags)))
    text_result_iter = []
    for sentence, sent_pos in zip(sentences, sentences_pos):
        pred_pos, pred_prob_log = viterbi(sentence, trans_probs, em_probs, start_token)
        pred_pos = [universal_pos_map[p] for p in pred_pos]
        viterbi_log_prob += pred_prob_log
        iteration_result.append(pred_pos)
        text_result_iter.append([(words[w], universal_pos_tags[p]) for w, p in zip(sentence, pred_pos)])

        for w, p in zip(sentence, pred_pos):
            if words[w] in prototypes:
                prototype_total += 1
                if universal_pos_tags[p] == prototypes[words[w]]:
                    prototype_correct += 1
        # print [(words[w], pos[p]) for w, p in zip(sentence, pred_pos)]
        # print pred_prob_log
        for pred, gold, word in zip(pred_pos, sent_pos, sentence):
            total += 1
            if pred == gold:
                correct += 1
            word_pos_counts[word, pred] += 1
    print name, correct / float(total)
    print name, 'viterbi log prob:', viterbi_log_prob

    print name, 'prototype accuracy', prototype_correct / float(prototype_total)

    word_pos_indicator = (word_pos_counts > 0)
    print name, 'average number of pos per word', word_pos_indicator.sum() / float(n_words)
    word_pos_counts /= word_pos_counts.sum(axis=1)[:, numpy.newaxis]
    print name, 'word counts', universal_pos_tags, word_pos_counts.sum(axis=0)

    return iteration_result, text_result_iter

def run(initial_lg_transition, initial_lg_emission, source_transition, emission_mask, universal_pos_map, name):
    prev_trans_weights = initial_lg_transition
    prev_em_weights = None
    trans_probs = probs_from_weights(initial_lg_transition)
    em_probs = probs_from_weights(initial_lg_emission, emission_mask)

    n_pos = initial_lg_transition.shape[0]

    start_token = universal_pos_map.index(universal_pos_tags.index('START'))

    for iter in xrange(iterations):
        trans_counts = numpy.zeros((n_pos, n_pos))
        em_counts = numpy.zeros((n_pos, len(words)))

        for sentence in sentences:
            add_counts_forward_backward(sentence, trans_probs, em_probs, start_token, trans_counts, em_counts)


        # normalize counts to get model params
        #trans_probs, em_probs = model_from_counts(trans_counts, em_counts, n_pos)

        trans_probs, em_probs, prev_trans_weights, prev_em_weights, result_description = \
            optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, source_transition,
                              update_transitions, direct_gradient, emission_feature_matrix, len(emission_feature_map.feats),
                              l2_weight_regulariztion_coeff, emission_mask)
        print name, "iteration:", iter, result_description

        if iter % 10 == 9:
            test(name, trans_probs, em_probs, start_token, universal_pos_map)


    iteration_result, text_result_iter = test(name, trans_probs, em_probs, start_token, universal_pos_map)

    print name, 'transition distance moved', numpy.abs(source_transition - trans_probs).sum()

    if out_file is not None:
        load_and_save.write_sentences_to_file(text_result_iter, out_file + '-' + name)

    return iteration_result

if random_init:
    n_pos = len(universal_pos_tags)
    td = numpy.ones((n_pos, n_pos))
    td /= n_pos
    other_langs_trans_dists.append(td)
    other_langs_tag_dists.append(None)
    universal_pos_maps.append(range(n_pos))
    annotated_languages.append('random')

initializations = []
for repeat_number in xrange(repeat):
    for trans_dist, tag_dist, pos_map, language_name in zip(other_langs_trans_dists, other_langs_tag_dists, universal_pos_maps, annotated_languages):
        n_pos = trans_dist.shape[0]
        initial_lg_trans = numpy.log(trans_dist)
        init_em = numpy.ones((n_pos, n_words))

        emission_mask = numpy.ones((n_pos, n_words))

        # set punctuation
        def is_punc(s):
            return sum(1 if l.isalnum() else 0 for l in s) == 0
        detected_punc = []
        for w in xrange(n_words):
            word_is_punc = is_punc(words[w])
            if word_is_punc:
                detected_punc.append(words[w])
            for p in xrange(n_pos):
                pos_is_punc = (universal_pos_tags[pos_map[p]] == '.')

                if pos_is_punc and not word_is_punc:
                    init_em[p, w] = .01
                    emission_mask[p,w] = 1e-12
                elif word_is_punc and not pos_is_punc:
                    init_em[p, w] = .01
                    emission_mask[p,w] = 1e-12

            if use_prototypes and words[w] in prototypes:
                proto_pos = universal_pos_tags.index(prototypes[words[w]])
                for p in xrange(n_pos):
                    if pos_map[p] != proto_pos:
                        init_em[p, w] = .0001
                        emission_mask[p,w] = 1e-12

        print 'detected punctuation', detected_punc

        init_em /= init_em.sum(axis=1)[:, numpy.newaxis]
        initial_lg_em = numpy.log(init_em)

        name = language_name + str(repeat_number)

        if initialization_noise_level > 0:
            initial_lg_trans += numpy.random.normal(scale=initialization_noise_level, size=initial_lg_trans.shape)
            initial_lg_em += numpy.random.normal(scale=initialization_noise_level, size=initial_lg_em.shape)

        initializations.append((initial_lg_trans, initial_lg_em, trans_dist, emission_mask, pos_map, name))

num_cores = min(len(initializations), max(multiprocessing.cpu_count() / 2, 1))
results = Parallel(n_jobs=num_cores)(delayed(run)(ilt, ile, tnsd, mask, pm, n) for ilt, ile, tnsd, mask, pm, n in initializations)
#results = [run(ilt, ile, tnsd, tgd, pm, n) for ilt, ile, tnsd, tgd, pm, n in initializations]

# vote on final output over different initializations
n_pos = len(universal_pos_tags)
universal_start_token = universal_pos_tags.index('START')
pred_sent_pos = []
out_text_sents = []
total = 0
correct = 0
confusion_matrix = numpy.zeros((n_pos, n_pos))

trans_counts = numpy.zeros((n_pos, n_pos))
em_counts = numpy.zeros((n_pos, n_words))
for s, sent_pos in enumerate(sentences_pos):
    pred_for_sent = []
    out_text_sent = []
    prev_pos = universal_start_token
    for w, pos in enumerate(sent_pos):
        votes = [results[i][s][w] for i in xrange(len(results))]
        pred = int(scipy.stats.mode(votes)[0])
        confusion_matrix[pred, pos] += 1
        pred_for_sent.append(pred)
        if pred == pos:
            correct += 1
        total += 1

        # do counts
        trans_counts[prev_pos, pred] += 1
        em_counts[pred, sentences[s][w]] += 1

        out_text_sent.append((text_sentences[s][w][0], universal_pos_tags[pred]))

    trans_counts[prev_pos, universal_start_token] += 1

    pred_sent_pos.append(pred_for_sent)
    out_text_sents.append(out_text_sent)
print 'score', correct / float(total)
# print confusion_matrix
print 'many to one', confusion_matrix.max(axis=1).sum() / float(confusion_matrix.sum())
if out_file is not None:
    load_and_save.write_sentences_to_file(out_text_sents, out_file)

'''
trans_counts += 1e-8
em_counts += 1e-8
voted_trans_probs = trans_counts / trans_counts.sum(axis=1)[:, numpy.newaxis]
voted_em_probs = em_counts / em_counts.sum(axis=1)[:, numpy.newaxis]
run(numpy.log(voted_trans_probs), numpy.log(voted_em_probs), voted_trans_probs, numpy.zeros(n_pos), range(n_pos), 'final')
'''