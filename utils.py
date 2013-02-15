
def MSE(predictions, labels):
    mse = 0
    for p, l in zip(predictions, labels):
        mse += pow(p-l,2)
    mse /= len(labels)
    return mse

def is_converged(prev_error, current_error, thresh):
    return (abs(prev_error - current_error) <= thresh)