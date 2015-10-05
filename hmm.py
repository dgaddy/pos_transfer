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
            for j in xrange(n):
                if j == start_token:
                    continue

                val = 0
                for k in xrange(n):
                    val += trans_probs[k, j] * lattice[i-1, k]
                val *= emission_probs[j, w]
                lattice[i, j] = val

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
        for j in xrange(n_pos):
            if j == start_token:
                continue

            val = 0
            for k in xrange(n_pos):
                val += trans_probs[j, k] * backward_lattice[i+1, k] * emission_probs[k, sentence_int[i+1]]
            val *= scaling_factors[i+1]
            backward_lattice[i, j] = val

    probs = forward_lattice * backward_lattice

    for i in xrange(len_sent):
        # emission counts
        for j in xrange(n_pos):
            em_counts[j, sentence_int[i]] += probs[i, j]
        # transition counts
        if i == 0:
            for j in xrange(n_pos):
                trans_counts[start_token, j] += probs[i, j]
        else:
            m = numpy.zeros((n_pos, n_pos))
            for j in xrange(n_pos):
                for k in xrange(n_pos):
                    m[j, k] = forward_lattice[i-1, j] * backward_lattice[i, k] * trans_probs[j, k] * emission_probs[k, sentence_int[i]]
            m /= m.sum()
            for j in xrange(n_pos):
                for k in xrange(n_pos):
                    trans_counts[j, k] += m[j, k]
        if i == len_sent - 1:
            for j in xrange(n_pos):
                trans_counts[j, start_token] += probs[i, j]


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

def get_shared_model(shared_feats, reg_coeff, languages, do_features, do_freq_feats, pos_tags):
    num_languages = len(languages)
    num_shared_feats = len(shared_feats)
    start_token = pos_tags.index('START')
    values = numpy.zeros((num_languages, num_shared_feats))
    transitions = []
    for l, language in enumerate(languages):
        text_sentences = load_and_save.read_sentences_from_file('pos_data/'+language+'.pos')

        sentences, words, sentences_pos, pos = load_and_save.integer_sentences(text_sentences, pos=pos_tags, max_words=1000)
        n_pos = len(pos_tags)

        trans_feats, trans_feat_map = make_transition_features(pos)
        em_feats, em_feat_map = make_emission_features(pos, words, sentences, do_freq_feats)
        num_trans_feats = len(trans_feat_map.feats)
        num_em_feats = len(em_feat_map.feats)

        trans_counts = numpy.zeros((n_pos, n_pos))
        em_counts = numpy.zeros((n_pos, len(words)))

        for sent, sent_pos in zip(sentences, sentences_pos):
            for i in xrange(len(sent)):
                w = sent[i]
                p = sent_pos[i]
                if i == 0:
                    trans_counts[start_token, p] += 1
                else:
                    trans_counts[sent_pos[i-1], p] += 1
                em_counts[p, w] += 1
            trans_counts[sent_pos[-1], start_token] += 1

        trans_counts += 1e-5
        trans_probs = numpy.nan_to_num(trans_counts / trans_counts.sum(axis=1).reshape((n_pos,1)))
        transitions.append(trans_probs)

        if do_features:
            _, _, tw, ew, _ = optimize_features(trans_counts, em_counts, None, None,
                                             numpy.zeros(num_trans_feats), numpy.zeros(num_em_feats),
                                             numpy.ones(num_trans_feats)*reg_coeff, numpy.ones(num_em_feats)*reg_coeff,
                                             trans_feats, em_feats, num_trans_feats, num_em_feats, 0)

            for i, feat in enumerate(shared_feats):
                if feat in trans_feat_map.map:
                    val = tw[trans_feat_map.map[feat]]
                elif feat in em_feat_map.map:
                    val = ew[em_feat_map.map[feat]]
                else:
                    raise Exception("Missing shared feature: " + feat)
                values[l, i] = val

    return values, transitions

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


def get_shared_features(pos, share_freq_feats):

    result = []
    for p1 in pos:
        if share_freq_feats:
            for n in xrange(5):
                result.append('emit_freq %s %d' % (p1, n))

        for p2 in pos:
            result.append('trans %s %s' % (p1, p2))

        for p in '.', ',', '?':
            result.append('emit %s %s' % (p1, p))
    return result


def make_transition_features(pos):
    n_pos = len(pos)
    fm = FeatureMap()
    feature_matrix = numpy.empty((n_pos, n_pos), dtype=object)
    for i in xrange(n_pos):
        for j in xrange(n_pos):
            feature_matrix[i, j] = [fm.get_id('trans %s %s' % (pos[i], pos[j]))]
    return feature_matrix, fm


def make_emission_features(pos, words, sentences_int, do_freq_feats):

    word_counts = Counter()
    for sentence in sentences_int:
        word_counts.update(sentence)
    total = float(sum(word_counts.values()))
    word_freq_bins = {w: int(-math.log(c/total, 10)) for w, c in word_counts.iteritems()}
    # print 'bin counts', Counter(word_freq_bins.values())

    n_pos = len(pos)
    n_words = len(words)
    fm = FeatureMap()
    feature_matrix = numpy.empty((n_pos, n_words), dtype=object)
    for i in xrange(n_pos):
        # make sure all the shared features are there
        if do_freq_feats:
            for n in xrange(5):
                fm.get_id('emit_freq %s %d' % (pos[i], n))
        for p in ',', '.', '?':
            fm.get_id('emit %s %s' % (pos[i], p))

        for j in xrange(n_words):
            word = words[j]
            feats = ['emit %s %s' % (pos[i], word), 'emit_suf %s %s' % (pos[i], word[-3:])]
            if do_freq_feats:
                feats.append('emit_freq %s %d' % (pos[i], word_freq_bins[j]))
            feature_matrix[i, j] = [fm.get_id(f) for f in feats]

    return feature_matrix, fm


def theta(w, decision_feat_array):
        n = decision_feat_array.size
        v = numpy.zeros(n)
        for d in xrange(n):
            v[d] = sum(w[f] for f in decision_feat_array[d])
        v -= v.max()
        exp = numpy.exp(v)
        sum_exp = exp.sum()
        return exp / sum_exp

def probs_from_weights(weights, feats):
    probs = numpy.zeros(feats.shape)
    for p in xrange(feats.shape[0]):
        probs[p, :] = theta(weights, feats[p, :])
    return probs

def optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, prior_trans_weights, prior_em_weights, trans_reg_coeffs, em_reg_coeffs, trans_feats, em_feats, num_trans_feats, num_em_feats, l1_reg_coeff):
    def l(weights, feat_map, counts, prior, reg_coeffs): # prior is shape (num_feats), reg_coeffs is in shape (num_feats)
        result = 0
        for p in xrange(counts.shape[0]):
            log_t = numpy.log(theta(weights, feat_map[p,:]))
            for w in xrange(counts.shape[1]):
                result += counts[p, w] * log_t[w]

        diff = weights - prior
        result -= (reg_coeffs * diff * diff).sum()

        if l1_reg_coeff > 0:
            # an l1 penalty on the unnormalized exponents
            v = 0
            for p in xrange(counts.shape[0]):
                for w in xrange(counts.shape[1]):
                    v += math.exp(sum(weights[f] for f in feat_map[p, w]))
            result -= v

        return result

    def dl(weights, feat_map, counts, n_feats, prior, reg_coeffs):
        result = numpy.zeros(n_feats)
        for p in xrange(counts.shape[0]):
            t = theta(weights, feat_map[p,:])
            sum_theta_f = numpy.zeros(n_feats)
            for w in xrange(counts.shape[1]):
                for feat in feat_map[p, w]:
                    sum_theta_f[feat] += t[w]
            for w in xrange(counts.shape[1]):
                count = counts[p, w]
                for feat in feat_map[p, w]:
                    result[feat] += count
            result -= sum_theta_f * counts[p,:].sum()

        diff = weights - prior
        result -= 2 * reg_coeffs * diff

        if l1_reg_coeff > 0:
            # the derivative of the l1 on unnormalized exp is just the sum of all the exponents where a feature is
            for p in xrange(counts.shape[0]):
                for w in xrange(counts.shape[1]):
                    e = math.exp(sum(weights[f] for f in feat_map[p, w]))
                    for f in feat_map[p, w]:
                        result[f] -= e

        return result

    if prev_trans_weights is None:
        prev_trans_weights = numpy.zeros(num_trans_feats)
    result = scipy.optimize.minimize(lambda w: -l(w, trans_feats, trans_counts, prior_trans_weights, trans_reg_coeffs), prev_trans_weights, method='L-BFGS-B', jac=lambda w: -dl(w, trans_feats, trans_counts, num_trans_feats, prior_trans_weights, trans_reg_coeffs))
    trans_weights = result.x
    trans_log_prob = result.fun

    if prev_em_weights is None:
        prev_em_weights = numpy.zeros(num_em_feats)
    result = scipy.optimize.minimize(lambda w: -l(w, em_feats, em_counts, prior_em_weights, em_reg_coeffs), prev_em_weights, method='L-BFGS-B', jac=lambda w: -dl(w, em_feats, em_counts, num_em_feats, prior_em_weights, em_reg_coeffs), options={'ftol':1e-5})
    emission_weights = result.x
    em_log_prob = result.fun

    description_string =  'log probability %s (%s %s)' % (em_log_prob + trans_log_prob, em_log_prob, trans_log_prob)

    trans_probs = probs_from_weights(trans_weights, trans_feats)

    emission_probs = probs_from_weights(emission_weights, em_feats)

    return trans_probs, emission_probs, trans_weights, emission_weights, description_string


def model_from_counts(transition_counts, emission_counts, num_pos):
    transition_counts += 1e-5
    transition_probs = numpy.nan_to_num(transition_counts / transition_counts.sum(axis=1).reshape((num_pos, 1)))

    emission_counts += 1e-5
    emission_probs = numpy.nan_to_num(emission_counts / emission_counts.sum(axis=1).reshape((num_pos, 1)))

    return transition_probs, emission_probs


default_args = {'reg_coeff': 1e-4, 'shared_reg_coeff': 1, 'iterations': 20, 'share_stat_features': 'False',
        'init_noise_level': 0, 'repeat': 1, 'do_freq_features': 'False',
        'vote_across_langs': 'False', 'l1_exp_reg_coeff': 0}

args = default_args.copy()
for arg in sys.argv[1:]:
    i = arg.index(':')
    key = arg[:i]
    val = arg[i+1:]
    args[key] = val

(base_reg_coeff, shared_reg_coeff, iterations,
                   initialization_noise_level, repeat, out_file, source, target, do_freq_feats,
                   vote_across_langs, l1_exp_reg_coeff) = \
(float(args['reg_coeff']), float(args['shared_reg_coeff']), int(args['iterations']),
               float(args['init_noise_level']), int(args['repeat']),
               args['out_file'] if 'out_file' in args else None, args['sources'].split(',') if 'sources' in args else None, args['target'],
               args['do_freq_features'] == 'True', args['vote_across_langs'] == 'True', float(args['l1_exp_reg_coeff']))

print 'regularization coeff', base_reg_coeff
print 'shared regularization coeff', shared_reg_coeff
print 'iterations', iterations
print 'initalization noise', initialization_noise_level
print 'repeats', repeat
print 'output file', out_file
print 'do frequency features', do_freq_feats
print 'vote across languages', vote_across_langs
print 'l1 exponent regularizaiton', l1_exp_reg_coeff

language = target
print 'to', language
text_sentences = load_and_save.read_sentences_from_file('pos_data/'+language+'.pos')

WALS_map = load_and_save.load_WALS_map('pos_data/WALS_map')
annotated_languages = source if source is not None else [l for l in WALS_map if l != target]
print 'from', annotated_languages
num_shared_langs = len(annotated_languages)

pos_tags = universal_pos_tags

shared_feats = get_shared_features(pos_tags, do_freq_feats)

sentences, words, sentences_pos, pos = load_and_save.integer_sentences(text_sentences, pos=universal_pos_tags, max_words=1000)
n_pos = len(pos_tags)

trans_feats, trans_feat_map = make_transition_features(pos)
em_feats, em_feat_map = make_emission_features(pos, words, sentences, do_freq_feats)
num_trans_feats = len(trans_feat_map.feats)
num_em_feats = len(em_feat_map.feats)
all_prior_trans_weights = numpy.zeros((num_shared_langs, num_trans_feats))
all_prior_em_weights = numpy.zeros((num_shared_langs, num_em_feats))
trans_reg_coeffs = numpy.ones(num_trans_feats) * base_reg_coeff
em_reg_coeffs = numpy.ones(num_em_feats) * base_reg_coeff


shared_feats_values, other_langs_trans_dists = get_shared_model(shared_feats, base_reg_coeff, annotated_languages, True, do_freq_feats, pos_tags)

print "finished loading source models"

for feat_num, feat in enumerate(shared_feats):
    if feat in trans_feat_map.map:
        i = trans_feat_map.map[feat]
        all_prior_trans_weights[:, i] = shared_feats_values[:, feat_num]
        trans_reg_coeffs[i] = shared_reg_coeff
    elif feat in em_feat_map.map:
        i = em_feat_map.map[feat]
        all_prior_em_weights[:, i] = shared_feats_values[:, feat_num]
        em_reg_coeffs[i] = shared_reg_coeff
    else:
        raise Exception()

if vote_across_langs:
    repeat = shared_feats_values.shape[0]

def run(prev_trans_weights, prev_em_weights, name):
    prior_trans_weights = prev_trans_weights.copy()
    prior_em_weights = prev_em_weights.copy()
    trans_probs = probs_from_weights(prev_trans_weights, trans_feats)
    em_probs = probs_from_weights(prev_em_weights, em_feats)

    start_token = pos_tags.index('START')

    for iter in xrange(iterations):
        trans_counts = numpy.zeros((n_pos, n_pos))
        em_counts = numpy.zeros((n_pos, len(words)))

        for sentence in sentences:
            add_counts_forward_backward(sentence, trans_probs, em_probs, start_token, trans_counts, em_counts)


        # normalize counts to get model params
        #trans_probs, em_probs = model_from_counts(trans_counts, em_counts, n_pos)

        trans_probs, em_probs, prev_trans_weights, prev_em_weights, result_description = \
            optimize_features(trans_counts, em_counts, prev_trans_weights, prev_em_weights, prior_trans_weights, prior_em_weights,
                              trans_reg_coeffs, em_reg_coeffs, trans_feats, em_feats, num_trans_feats, num_em_feats, l1_exp_reg_coeff)
        print name, "iteration:", iter, result_description

    correct = 0
    total = 0
    iteration_result = []
    viterbi_log_prob = 0
    word_pos_counts = numpy.zeros((len(words), n_pos))
    text_result_iter = []
    for sentence, sent_pos in zip(sentences, sentences_pos):
        pred_pos, pred_prob_log = viterbi(sentence, trans_probs, em_probs, start_token)
        viterbi_log_prob += pred_prob_log
        iteration_result.append(pred_pos)
        text_result_iter.append([(words[w], pos_tags[p]) for w, p in zip(sentence, pred_pos)])
        # print [(words[w], pos[p]) for w, p in zip(sentence, pred_pos)]
        # print pred_prob_log
        for pred, gold, word in zip(pred_pos, sent_pos, sentence):
            total += 1
            if pred == gold:
                correct += 1
            word_pos_counts[word, pred] += 1
    print name, correct / float(total)
    print name, 'viterbi log prob:', viterbi_log_prob

    word_pos_counts /= (word_pos_counts.sum(axis=1).reshape((len(words), 1)))
    print 'word counts', universal_pos_tags, word_pos_counts.sum(axis=0)

    if out_file is not None:
        load_and_save.write_sentences_to_file(text_result_iter, out_file + '-' + name)

    return iteration_result

initializations = []
for repeat_number in xrange(repeat):

    if vote_across_langs:
        prev_trans_weights = all_prior_trans_weights[repeat_number, :].copy()
        prev_em_weights = all_prior_em_weights[repeat_number, :].copy()
        name = annotated_languages[repeat_number]
    else:
        prev_trans_weights = numpy.average(all_prior_trans_weights, axis=0)
        prev_em_weights = numpy.average(all_prior_em_weights, axis=0)
        name = 'init' + str(repeat_number)

    if initialization_noise_level > 0:
        prev_trans_weights += numpy.random.normal(scale=initialization_noise_level, size=(all_prior_trans_weights.shape[1]))
        prev_em_weights += numpy.random.normal(scale=initialization_noise_level, size=(all_prior_em_weights.shape[1]))

    initializations.append((prev_trans_weights, prev_em_weights, name))

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores/2)(delayed(run)(pt, pe, n) for pt, pe, n in initializations)
# results = [run(pt, pe, n) for pt, pe, n in initializations]

# vote on final output over different initializations
pred_sent_pos = []
out_text_sents = []
total = 0
correct = 0
confusion_matrix = numpy.zeros((n_pos, n_pos))
for s, sent_pos in enumerate(sentences_pos):
    pred_for_sent = []
    out_text_sent = []
    for w, pos in enumerate(sent_pos):
        votes = [results[i][s][w] for i in xrange(len(results))]
        pred = int(scipy.stats.mode(votes)[0])
        confusion_matrix[pred, pos] += 1
        pred_for_sent.append(pred)
        if pred == pos:
            correct += 1
        total += 1

        out_text_sent.append((text_sentences[s][w][0], pos_tags[pred]))
    pred_sent_pos.append(pred_for_sent)
    out_text_sents.append(out_text_sent)
print 'score', correct / float(total)
# print confusion_matrix
print 'many to one', confusion_matrix.max(axis=1).sum() / float(confusion_matrix.sum())
if out_file is not None:
    load_and_save.write_sentences_to_file(out_text_sents, out_file)
