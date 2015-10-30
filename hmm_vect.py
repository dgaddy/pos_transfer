import word_vect_loader
import load_and_save
import numpy
import math
import scipy.optimize
import numpy.linalg

universal_pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', 'START']

def theta(weights, word_vectors):
    weighted = word_vectors.dot(weights)
    weighted -= weighted.max() # so all exp values less than 1
    exp = numpy.exp(weighted)
    sum_exp = exp.sum()
    assert exp.shape == (word_vectors.shape[0],)
    return exp / sum_exp

def probs_from_weights(weights, word_vectors):
    # weights is shape (n_pos, n_feats)
    probs = numpy.zeros((weights.shape[0], word_vectors.shape[0]))
    for p in xrange(weights.shape[0]):
        probs[p, :] = theta(weights[p,:], word_vectors)
    return probs

def optimize_features(em_counts, em_reg_coeff, word_vectors, prev_em_weights=None):
    n_pos = em_counts.shape[0]
    vect_len = word_vectors.shape[1]
    def l(weights, counts, reg_coeff): # prior is shape (num_feats), reg_coeffs is in shape (num_feats)
        weights = weights.reshape((n_pos, vect_len))
        result = 0
        for p in xrange(counts.shape[0]):
            log_t = numpy.log(theta(weights[p,:], word_vectors))
            result += counts[p,:].dot(log_t)

        result -= reg_coeff * (weights * weights).sum()

        return result

    def dl(weights, counts, reg_coeff):
        weights = weights.reshape((n_pos, vect_len))

        result = numpy.zeros(weights.shape)
        for p in xrange(counts.shape[0]):
            actual_counts = word_vectors.T.dot(counts[p,:])
            t = theta(weights[p,:], word_vectors)
            expected_counts = word_vectors.T.dot(t) * counts[p,:].sum()
            result[p,:] = actual_counts - expected_counts

        result -= 2 * reg_coeff * weights

        return result.ravel()

    if prev_em_weights is None:
        prev_em_weights = numpy.zeros((n_pos, vect_len))
    result = scipy.optimize.minimize(lambda w: -l(w, em_counts, em_reg_coeff), prev_em_weights, method='L-BFGS-B', jac=lambda w: -dl(w, em_counts, em_reg_coeff), options={'ftol':1e-5})
    emission_weights = result.x.reshape((n_pos, vect_len))
    em_log_prob = result.fun

    description_string =  'log probability %s' % (em_log_prob)

    emission_probs = probs_from_weights(emission_weights, word_vectors)

    return emission_probs, emission_weights, description_string

def do_counts(sentences, sentences_pos, n_pos, n_words, start_token):
    trans_counts = numpy.zeros((n_pos, n_pos))
    em_counts = numpy.zeros((n_pos, n_words))

    for sent, sent_pos in zip(sentences, sentences_pos):
        for i in xrange(len(sent)):
            w = sent[i]
            p = sent_pos[i]
            em_counts[p, w] += 1
            if i == 0:
                trans_counts[start_token, p] += 1
            else:
                trans_counts[sent_pos[i-1], p] += 1
        trans_counts[sent_pos[-1], start_token] += 1

    return trans_counts, em_counts

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

def find_rotation(from_word_vects, to_word_vects):
    # based on http://nghiaho.com/?page_id=671
    H = from_word_vects.T.dot(to_word_vects)
    U,s,Vt = numpy.linalg.svd(H)
    rot =  Vt.T.dot(U.T)
    if numpy.linalg.det(rot) < 0:
        print 'reflecting'
        rot[-1,:] *= -1
    return rot

def find_linear_transformation(from_word_vects, to_word_vects):
    # does least squares for each dimension of to_vects
    result = numpy.zeros((from_word_vects.shape[1], to_word_vects.shape[1]))
    for i in xrange(to_word_vects.shape[1]):
        y = to_word_vects[:,i]
        X = from_word_vects
        b = numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        result[i,:] = b
    return result

def test(sentences, sentences_pos, name, trans_probs, em_probs, start_token):
    correct = 0
    total = 0
    iteration_result = []
    viterbi_log_prob = 0
    for sentence, sent_pos in zip(sentences, sentences_pos):
        pred_pos, pred_prob_log = viterbi(sentence, trans_probs, em_probs, start_token)
        viterbi_log_prob += pred_prob_log
        iteration_result.append(pred_pos)

        # print [(words[w], pos[p]) for w, p in zip(sentence, pred_pos)]
        # print pred_prob_log
        for pred, gold, word in zip(pred_pos, sent_pos, sentence):
            total += 1
            if pred == gold:
                correct += 1
    print name, correct / float(total)
    print name, 'viterbi log prob:', viterbi_log_prob

    return iteration_result

def make_vector_matrix(words_list, vector_map, vector_size):
    n_words = len(words_list)
    word_vect_matrix = numpy.zeros((n_words, vector_size))
    num_no_vect = 0
    for w in xrange(n_words):
        if words_list[w] in vector_map:
            v = vector_map[words_list[w]]
            v /= math.sqrt(v.dot(v)) # hurts performance slightly on monolingual
            word_vect_matrix[w] = v
        else:
            num_no_vect += 1
    print '%s words have no vector' % num_no_vect
    return word_vect_matrix


source_language = 'spanish'
target_language = 'english07'
pair_filename = 'word_pairs/es-en.pair'
reverse_pair = False

target_vectors, target_vect_size = word_vect_loader.load('pos_data/'+target_language+'.train.sent.vec')
source_vectors, source_vect_size = word_vect_loader.load('pos_data/'+source_language+'.train.sent.vec')
assert source_vect_size == target_vect_size
vect_size = source_vect_size

print 'loading'
source_text_sentences = load_and_save.read_sentences_from_file('pos_data/conll-'+source_language+'.pos')
source_sentences, source_words, source_sentences_pos, _ = load_and_save.integer_sentences(source_text_sentences, pos=universal_pos_tags, max_words=10000)
source_test_sentences = load_and_save.read_sentences_from_file('pos_data/conll-'+source_language+'-test.pos')
source_test_sentences, _, source_test_sentences_pos, _ = load_and_save.integer_sentences(source_test_sentences, pos=universal_pos_tags, words=source_words)

target_text_sentences = load_and_save.read_sentences_from_file('pos_data/conll-'+target_language+'-test.pos')
target_sentences, target_words, target_sentences_pos, _ = load_and_save.integer_sentences(target_text_sentences, pos=universal_pos_tags, max_words=10000)

source_vector_matrix = make_vector_matrix(source_words, source_vectors, vect_size)
target_vector_matrix = make_vector_matrix(target_words, target_vectors, vect_size)

print 'finding rotation'
translation_pairs = []
pair_file = open(pair_filename)
for line in pair_file:
    split = line.split()
    if split[0] in source_words and split[2] in target_words:
        translation_pairs.append((split[2], split[0]) if reverse_pair else (split[0], split[2]))

print 'using %s translation pairs' % len(translation_pairs)
source_trans_matrix = numpy.zeros((len(translation_pairs), vect_size))
target_trans_matrix = numpy.zeros((len(translation_pairs), vect_size))

for i, (sw, tw) in enumerate(translation_pairs):
    swi = source_words.index(sw)
    twi = target_words.index(tw)
    source_trans_matrix[i, :] = source_vector_matrix[swi, :]
    target_trans_matrix[i, :] = target_vector_matrix[twi, :]

#transform = find_rotation(target_trans_matrix, source_trans_matrix)
transform = find_linear_transformation(target_trans_matrix, source_trans_matrix)

source_reconstruction = target_trans_matrix.dot(transform.T)
reconstruction_error = source_reconstruction - source_trans_matrix
print 'average reconstruction error', numpy.sqrt((reconstruction_error * reconstruction_error).sum(axis=1)).mean()

print 'counting'
trans_counts, em_counts = do_counts(source_sentences, source_sentences_pos, len(universal_pos_tags), len(source_words), universal_pos_tags.index('START'))
trans_counts += 1e-5
trans_probs = trans_counts / trans_counts.sum(axis=1)[:, numpy.newaxis]

print 'optimizing emissions'
source_em_probs, weights, s = optimize_features(em_counts, 1e-6, source_vector_matrix)
em_counts += 1e-5
em_probs_from_counts = em_counts / em_counts.sum(axis=1)[:, numpy.newaxis]

translated_target_vector_matrix = target_vector_matrix.dot(transform.T)
target_em_probs = probs_from_weights(weights, translated_target_vector_matrix)

print 'testing'

start = universal_pos_tags.index('START')
test(source_test_sentences, source_test_sentences_pos, 'source embedding', trans_probs, source_em_probs, start)
test(source_test_sentences, source_test_sentences_pos, 'source count', trans_probs, em_probs_from_counts, start)
test(target_sentences, target_sentences_pos, 'target', trans_probs, target_em_probs, start)
