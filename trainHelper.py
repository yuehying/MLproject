import numpy as np

def batch_gen(all_train_data, batchsize):
    '''
    generate non-overlapping batch for training
    '''
    num_data = all_train_data.shape[0]
    idx = np.array(list(range(num_data)))
    np.random.shuffle(idx)
    remain = idx.shape[0]
    list_of_batchidx = []
    while remain > batchsize:
        range_start = num_data - remain
        list_of_batchidx.append([idx[range_start:range_start+batchsize]])
        remain = remain - batchsize
    return list_of_batchidx

def write_accuracy_output(accuracy, filename):
    # accuracy: list of float
    with open(filename, 'w') as file:
        for acc in accuracy:
            file.write("%1.4f," % acc)


def write_pred_output(predicted, filename='pred.csv'):
    entry_cnt = 0
    with open(filename, 'w') as file:
        file.write('Id,IsBrazilian\n')
        for pred in predicted:
            if entry_cnt+1 == len(predicted):
                towrite = str(entry_cnt) +',' + str(pred)
            else:
                towrite = str(entry_cnt) +',' +str(pred) + '\n'
            entry_cnt = entry_cnt + 1
            file.write(towrite)