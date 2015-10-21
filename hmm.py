from collections import Counter
import numpy
import scipy.optimize
import math
import sys
import scipy.stats
import load_and_save

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

def probs_from_weights(weights):
    # TODO: do we want to offset weights before exp for numerical reasons?

    exp_weights = numpy.exp(weights)
    return exp_weights / exp_weights.sum(axis=1)[:, numpy.newaxis]

def optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, source_trans_probs, source_tag_probs,
                      update_transitions, direct_gradient,
                      l2_weight_regularization, l0_coeff):

    norm_l = .1
    log_source_tag_probs = numpy.log(source_tag_probs + 1e-8)

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

    def loss_em(weights):
        weights = weights.reshape(em_counts.shape)
        p = probs_from_weights(weights)
        log_p = numpy.log(p)
        result = -((em_counts * log_p).sum())

        result += (weights * weights).sum() * l2_weight_regularization

        return result

    def d_loss_em(weights):
        weights = weights.reshape(em_counts.shape)
        p = probs_from_weights(weights)
        pos_counts = em_counts.sum(axis=1)
        result = (p * pos_counts[:, numpy.newaxis]) - em_counts

        result += 2 * l2_weight_regularization * weights

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
        prev_em_weights = numpy.zeros(em_counts.shape)
    result = scipy.optimize.minimize(loss_em, prev_em_weights, method='L-BFGS-B', jac=d_loss_em,
                                     options=({'maxiter':1} if direct_gradient else {}))
    emission_weights = result.x.reshape(em_counts.shape)
    em_log_prob = result.fun

    # sketchy way of trying to enforce l0
    sorted_weights = numpy.sort(emission_weights, axis=1)
    cutoffs = sorted_weights[:,950]
    emission_weights[universal_pos_closed, :] -= (emission_weights[universal_pos_closed,:] < cutoffs[universal_pos_closed][:,numpy.newaxis]) * l0_coeff

    emission_probs = probs_from_weights(emission_weights)

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
                'prototypes': 'False'}

args = default_args.copy()
for arg in sys.argv[1:]:
    i = arg.index(':')
    key = arg[:i]
    val = arg[i+1:]
    args[key] = val

(iterations, initialization_noise_level, repeat, out_file, source, target, update_transitions,
 l2_weight_regulariztion_coeff, l0_coefficient,
 direct_gradient, do_language_specific_tags, use_prototypes) = \
(int(args['iterations']), float(args['init_noise_level']), int(args['repeat']),
               args['out_file'] if 'out_file' in args else None, args['sources'].split(',') if 'sources' in args else None, args['target'],
               args['update_transitions'] == 'True', float(args['reg_coeff']), float(args['l0_coeff']),
               args['direct_gradient'] == 'True', args['do_lang_specific'] == 'True', args['prototypes'] == 'True')

print 'iterations', iterations
print 'initalization noise', initialization_noise_level
print 'repeats', repeat
print 'output file', out_file
print 'update transitions', update_transitions
print 'l2 weight regularization', l2_weight_regulariztion_coeff
print 'l0 coefficient', l0_coefficient
print 'direct gradient', direct_gradient
print 'language specific tags', do_language_specific_tags


prototypes = {'the':'DET', 'are':'VERB', 'said':'VERB', 'and':'CONJ'}

print 'prototypes', use_prototypes, prototypes

language = target
print 'to', language
text_sentences = load_and_save.read_sentences_from_file('pos_data/'+language+'.pos')
sentences, words, sentences_pos, pos = load_and_save.integer_sentences(text_sentences, pos=universal_pos_tags, max_words=1000)
n_words = len(words)

WALS_map = load_and_save.load_WALS_map('pos_data/WALS_map')
annotated_languages = source if source is not None else [l for l in WALS_map if l != target]
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

def run(initial_lg_transition, initial_lg_emission, source_transition, source_tag_distribution, universal_pos_map, name):
    prev_trans_weights = initial_lg_transition
    prev_em_weights = initial_lg_emission
    trans_probs = probs_from_weights(initial_lg_transition)
    em_probs = probs_from_weights(initial_lg_emission)

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
            optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, source_transition, source_tag_distribution,
                              update_transitions, direct_gradient,
                              l2_weight_regulariztion_coeff, l0_coefficient)
        print name, "iteration:", iter, result_description

        if iter % 10 == 9:
            test(name, trans_probs, em_probs, start_token, universal_pos_map)


    iteration_result, text_result_iter = test(name, trans_probs, em_probs, start_token, universal_pos_map)

    print name, 'transition distance moved', numpy.abs(source_transition - trans_probs).sum()

    if out_file is not None:
        load_and_save.write_sentences_to_file(text_result_iter, out_file + '-' + name)

    return iteration_result

initializations = []
for repeat_number in xrange(repeat):
    for trans_dist, tag_dist, pos_map, language_name in zip(other_langs_trans_dists, other_langs_tag_dists, universal_pos_maps, annotated_languages):
        n_pos = trans_dist.shape[0]
        initial_lg_trans = numpy.log(trans_dist)
        init_em = numpy.ones((n_pos, n_words))

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
                elif word_is_punc and not pos_is_punc:
                    init_em[p, w] = .01

            if use_prototypes and words[w] in prototypes:
                proto_pos = universal_pos_tags.index(prototypes[words[w]])
                for p in xrange(n_pos):
                    if pos_map[p] != proto_pos:
                        init_em[p, w] = .01

        print 'detected punctuation', detected_punc

        init_em /= init_em.sum(axis=1)[:, numpy.newaxis]
        initial_lg_em = numpy.log(init_em)

        name = language_name + str(repeat_number)

        if initialization_noise_level > 0:
            initial_lg_trans += numpy.random.normal(scale=initialization_noise_level, size=initial_lg_trans.shape)
            initial_lg_em += numpy.random.normal(scale=initialization_noise_level, size=initial_lg_em.shape)

        initializations.append((initial_lg_trans, initial_lg_em, trans_dist, tag_dist, pos_map, name))

num_cores = min(len(initializations), max(multiprocessing.cpu_count() / 2, 1))
results = Parallel(n_jobs=num_cores)(delayed(run)(ilt, ile, tnsd, tgd, pm, n) for ilt, ile, tnsd, tgd, pm, n in initializations)
# results = [run(ilt, ile, td, n) for pt, pe, n in initializations]

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