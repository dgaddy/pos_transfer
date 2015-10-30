import numpy
from collections import Counter


def get_word_context_vectors(sentences_int, n_words):
    counter = Counter(w for s in sentences_int for w in s)
    most_common_words = [w for w, _ in counter.most_common(499)]
    # going to use the most common 499 words + the start/end token
    # index 0 for start/end and the rest for the common words
    most_common_map = {w: i+1 for i, w in enumerate(most_common_words)}

    # the second column is an indicator vecfor the most common 500 words in position -2, -1, +1, +2
    full_context_matrix = numpy.zeros((n_words, 500*4))
    for sentence in sentences_int:
        for i, w in enumerate(sentence):
            if i > 0:
                wp = sentence[i-1]
                if wp in most_common_map:
                    full_context_matrix[w, most_common_map[wp]] += 1
            if i > 1:
                wpp = sentence[i-2]
                if wpp in most_common_map:
                    full_context_matrix[w, most_common_map[wpp] + 500] += 1
            if i < len(sentence)-1:
                wn = sentence[i+1]
                if wn in most_common_map:
                    full_context_matrix[w, most_common_map[wn] + 1000] += 1
            if i < len(sentence)-2:
                wnn = sentence[i+2]
                if wnn in most_common_map:
                    full_context_matrix[w, most_common_map[wnn] + 1500] += 1
        # adding counts for start/end token
        full_context_matrix[sentence[0], 0] += 1 # prev has no offset
        full_context_matrix[sentence[-1], 1000] += 1 # next has offset 1000
        if len(sentence) > 1:
            full_context_matrix[sentence[1], 500] += 1
            full_context_matrix[sentence[-2], 1500] += 1


    u,s,v = numpy.linalg.svd(full_context_matrix)
    word_vects = u[:, :250]
    word_vects /= numpy.sqrt((word_vects*word_vects).sum(axis=1)[:,numpy.newaxis]) # renormalize
    return word_vects

def get_prototypes(words, vectors, prototype_ids):
    n_words = len(words)
    prototypes_for_words = [[] for _ in xrange(n_words)]
    for w1 in xrange(n_words):
        for prototype in prototype_ids:
            similarity = vectors[w1,:].dot(vectors[prototype, :])
            if similarity > .35:
                prototypes_for_words[w1].append(words[prototype])
    return prototypes_for_words

if __name__ == "__main__":
    text_filename = 'pos_data/conll-english07.pos'
    out_filename = 'context_svd_output.vec'

    counter = Counter()
    with open(text_filename, 'r') as file:
        for line in file:
            counter.update(line.split())

    # the words used in context vectors
    most_common_words = [w for w, _ in counter.most_common(499)]
    # going to use the most common 499 words + the start/end token
    # index 0 for start/end and the rest for the common words
    most_common_map = {w: i+1 for i, w in enumerate(most_common_words)}

    # the words we are getting vectors for
    words = [w for w, _ in counter.most_common(10000)]
    word_map = {w: i for i, w in enumerate(words)}

    # the second column is an indicator vecfor the most common 500 words in position -2, -1, +1, +2
    full_context_matrix = numpy.zeros((len(words), 500*4))
    with open(text_filename, 'r') as file:
        for line in file:
            sentence = line.split()
            for i, word in enumerate(sentence):
                if word not in word_map:
                    continue
                w = word_map[word]
                if i > 0:
                    wp = sentence[i-1]
                    if wp in most_common_map:
                        full_context_matrix[w, most_common_map[wp]] += 1
                else:
                    full_context_matrix[w, 0] += 1 # start token is 0
                if i > 1:
                    wpp = sentence[i-2]
                    if wpp in most_common_map:
                        full_context_matrix[w, most_common_map[wpp] + 500] += 1
                else:
                    full_context_matrix[w, 500] += 1
                if i < len(sentence)-1:
                    wn = sentence[i+1]
                    if wn in most_common_map:
                        full_context_matrix[w, most_common_map[wn] + 1000] += 1
                else:
                    full_context_matrix[w, 1000] += 1
                if i < len(sentence)-2:
                    wnn = sentence[i+2]
                    if wnn in most_common_map:
                        full_context_matrix[w, most_common_map[wnn] + 1500] += 1
                else:
                    full_context_matrix[w, 1500] += 1


    u,s,v = numpy.linalg.svd(full_context_matrix)
    word_vects = u[:, :250]
    word_vects /= numpy.sqrt((word_vects*word_vects).sum(axis=1)[:,numpy.newaxis]) # renormalize

    with open(out_filename, 'w') as file:
        file.write('%s %s\n' % (len(words), 250))
        for w in xrange(len(words)):
            file.write(words[w])
            file.write(' ')
            for v in word_vects[w,:]:
                file.write(str(v))
                file.write(' ')
            file.write('\n')
