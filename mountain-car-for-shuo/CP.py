import numpy as np
import utils

np.random.seed(42)

def compare(ar, thresh):
    results = []
    for i in range(ar.shape[1]):
        results.append(ar)

epsilon = 0.1
delta = 0.1

true_states = np.loadtxt("true_state.csv", delimiter=",")
pred_states = np.loadtxt("pred_state.csv", delimiter=",")

indices = np.arange(true_states.shape[0])
np.random.shuffle(indices)
true_states = true_states[indices, ]
pred_states = pred_states[indices, ]
true_cal = true_states[:int(true_states.shape[0]*0.5),]
pred_cal = pred_states[:int(pred_states.shape[0]*0.5),]
true_test = true_states[int(true_states.shape[0]*0.5):,]
pred_test = pred_states[int(pred_states.shape[0]*0.5):,]
res_cal = np.abs(true_cal - pred_cal)

epsilon = utils.find_maximum_train_error_allow(epsilon, delta, n=res_cal.shape[0])

quantile_x = np.quantile(res_cal[:,0], 1-epsilon)
quantile_v = np.quantile(res_cal[:,1], 1-epsilon)
quantile_c = np.quantile(res_cal[:,2], 1-epsilon)
quantile_d = np.quantile(res_cal[:,3], 1-epsilon)

count_x = np.sum((true_test[:,0] >= pred_test[:,0] - quantile_x) * (true_test[:,0] <= pred_test[:,0] + quantile_x))
count_v = np.sum((true_test[:,1] >= pred_test[:,1] - quantile_v) * (true_test[:,1] <= pred_test[:,1] + quantile_v))
count_c = np.sum((true_test[:,2] >= pred_test[:,2] - quantile_c) * (true_test[:,2] <= pred_test[:,2] + quantile_c))
count_d = np.sum((true_test[:,3] >= pred_test[:,3] - quantile_d) * (true_test[:,3] <= pred_test[:,3] + quantile_d))

rate_x = count_x / true_test.shape[0]
rate_v = count_v / true_test.shape[0]
rate_c = count_c / true_test.shape[0]
rate_d = count_d / true_test.shape[0]

print("X coverage:", rate_x)
print("V coverage:", rate_v)
print("C coverage:", rate_c)
print("D coverage:", rate_d)

print("X range:", quantile_x)
print("V range:", quantile_v)
print("C range:", quantile_c)
print("D range:", quantile_d)
