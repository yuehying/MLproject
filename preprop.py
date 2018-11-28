import numpy as np

def decoding(file):
    '''
    :param file: original csv file
    :return: list of [first_name, last_name, label]
    '''
    file_unicode = []
    for line in file:
        token = line.rstrip().lower().split(',')

        curr_line_tokenlist = []
        tmp = None
        if not str.isnumeric(token[0]): # first line, specifying field property
            continue
        else :
            cmd = 'b\'' + token[1] + '\''+ '.decode(\'utf-8\')'
            first_name_decoded = eval(cmd)
            cmd = 'b\'' + token[2] + '\'' + '.decode(\'utf-8\')'
            last_name_decoded = eval(cmd)
            file_unicode.append([first_name_decoded, last_name_decoded, token[3]])

    return file_unicode

def getLabel(file):
    label = (np.array(file)[:, 2]).tolist()
    label = [1 if gold=='1' else 0 for gold in label]
    return label

def genNgram(file, N=1):
    '''
    :param file: decoded file
    :param N: 1=unigram, 2=bigram, 3=trigram
    :return:  list of (list of tokenized N-gram)
    '''
    n_gram_list = []

    # convert in form $first name$+last name+
    for line in file:

        curr_line_tokenlist = []
        tmp = '${0}$+{1}+'.format(line[0], line[1])

        if not tmp==None:
            for idx in range(len(tmp)-N+1):
                curr_line_tokenlist.append(tmp[idx:idx+N])
            if not curr_line_tokenlist==[]:
                n_gram_list.append(curr_line_tokenlist)

    return n_gram_list

def ngram2embedded(ngram, w2vmodel):
    # convert alphabetical ngram into numerical ngram
    ngram_embedded = []
    for name in ngram:
        buffer = []
        for gram in name:
            buffer.append(w2vmodel.wv[gram])
        ngram_embedded.append(buffer)
    return ngram_embedded

def ngram2padded(ngram, max_timestep):
    '''
    pad every entry in a gram to its max length  with 0s
    note 0s are 0 vectors
    eg: embed dim=3, current gram timestep=1, pad to timestep 3
    [ [x,y,z], [0,0,0], [0,0,0]  ]
    '''
    emb_dimension = len(ngram[0][0])
    pad_vec = np.array(emb_dimension*[0.0], dtype=np.float32)
    ngram_padded = []
    for grams in ngram:
        buffer = [grams[itr] if itr<len(grams) else pad_vec for itr in range(max_timestep)]
        ngram_padded.append(buffer)
    ngram_padded = np.array(ngram_padded)
    return ngram_padded