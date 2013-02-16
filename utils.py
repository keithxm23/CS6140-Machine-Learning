
def MSE(predictions, labels):
    mse = 0
    for p, l in zip(predictions, labels):
        mse += pow(p-l,2)
    mse /= len(labels)
    return mse

def is_converged(prev_error_arr, current_error_arr, thresh):
    return ((sum(prev_error_arr) - sum(current_error_arr)) <= thresh)

def ABSE(predictions, labels):#absoulte error
    err = 0
    for i in xrange(len(labels)):
        err += abs(predictions[i]-labels[i])
    err /= len(labels)
    return err