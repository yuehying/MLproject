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