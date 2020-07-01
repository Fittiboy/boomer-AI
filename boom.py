import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def relu_deriv(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_deriv(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def initialize(dims, method="none"):
    factors = {"none": 1}
    params = {}
    for l in range(1, len(dims)):
        factors["he"] = np.sqrt(2 / dims[l-1])
        factors["xavier"] = np.sqrt(1 / dims[l-1])
        
        W = np.random.randn(dims[l], dims[l-1])
        params["W" + str(l)] = np.multiply(W, factors[method])
        params["b" + str(l)] = np.zeros((dims[l], 1))
        
    return params

def forward_prop(X, params, activation="relu", last="sigmoid"):
    activations = {
        "relu": relu,
        "sigmoid": sigmoid
    }
    layers = len(params) // 2
    caches = []
    A = X
    for l in range(1, layers):
        W = params["W" + str(l)]
        b = params["b" + str(l)]
        Z = np.dot(W, A) + b
        
        caches.append({
            "W": W,
            "A": A,
            "Z": Z,
            "b": b
        })
        A = activations[activation](Z)

    W = params["W" + str(l+1)]
    b = params["b" + str(l+1)]
    Z = np.dot(W, A) + b
    caches.append({
        "W": W,
        "A": A,
        "Z": Z,
        "b": b
    })
    A = activations[last](Z)
    return A, caches

def compute_cost(Y, A, params=None, method="none", lambd=0, last="sigmoid"):
    extra_terms = {
        "none": 0,
        "l2": 0 
    }
    m = Y.shape[1]
    if method == "l2":
        layers = len(params) // 2
        W = [params["W" + str(l)] for l in range(1, layers + 1)]
        W = [np.sum(np.square(w)) for w in W]
        W = np.sum(W)
        extra_terms["l2"] = (lambd / (2 * m)) * W
    if last == "sigmoid":
        first_term = np.multiply(np.log(A), Y)
        second_term = np.multiply(np.log(1 - A), 1 - Y)
        cost = ((-1 / m) * np.nansum(first_term + second_term)) + extra_terms[method]
    elif last == "relu":
        cost = (1 / m) * np.sum(np.abs(A - Y)) + extra_terms[method]
    return cost

def back_prop(A, caches, Y, activation="relu",
              method="none", lambd=0, last="sigmoid",
              beta_1=0, beta_2=0, grads_old={}, iter_num=1):
    extra_terms = {
        "none": 0,
        "l2": lambd / Y.shape[1]
    }
    derivs = {
        "relu": relu_deriv,
        "sigmoid": sigmoid_deriv
    }
    m = Y.shape[1]
    layers = len(caches)
    activ = "sigmoid"
    grads = {}
    if last == "sigmoid":
        dA = np.divide(1 - Y,  1 - A) - np.divide(Y, A)
    elif last == "relu":
        dA = np.array([1 if A[0][i] > Y[0][i] else -1 for i in range(len(A[0]))])
        dA = dA.reshape(1, -1)
    for l in reversed(range(1, layers + 1)):
        cache = caches.pop()
        W = cache["W"]
        Z = cache["Z"]
        A = cache["A"]
        dZ = derivs[activ](dA, Z)
        activ = activation
        
        VdW = grads_old["VdW" + str(l)]
        Vdb = grads_old["Vdb" + str(l)]
        SdW = grads_old["SdW" + str(l)]
        Sdb = grads_old["Sdb" + str(l)]
        
        dW = (1 / m) * np.dot(dZ, A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dW += extra_terms[method] * W
        
        VdW = beta_1 * VdW + (1 - beta_1) * dW
        Vdb = beta_1 * Vdb + (1 - beta_1) * db
        
        VdWc = VdW / (1 - beta_1**iter_num)
        Vdbc = Vdb / (1 - beta_1**iter_num)
        
        SdW = beta_2 * SdW + (1 - beta_2) * dW**2
        Sdb = beta_2 * Sdb + (1 - beta_2) * db**2
        
        SdWc = SdW / (1 - beta_2**iter_num)
        Sdbc = Sdb / (1 - beta_2**iter_num)
        
        grads["VdW" + str(l)] = VdW
        grads["Vdb" + str(l)] = Vdb
        grads["SdW" + str(l)] = SdW
        grads["Sdb" + str(l)] = Sdb
        
        grads["VdWc" + str(l)] = VdWc
        grads["Vdbc" + str(l)] = Vdbc
        grads["SdWc" + str(l)] = SdWc
        grads["Sdbc" + str(l)] = Sdbc
        
        dA = np.dot(W.T, dZ)
    
    return grads

def optimize(params, grads, learn_rate):
    layers = len(params) // 2
    for l in range(1, layers + 1):
        VdW = grads["VdWc" + str(l)]
        Vdb = grads["Vdbc" + str(l)]
        SdW = grads["SdWc" + str(l)]
        Sdb = grads["Sdbc" + str(l)]
        W_adjust = VdW / (np.sqrt(SdW) + 10e-8)
        b_adjust = Vdb / (np.sqrt(Sdb) + 10e-8)
        params["W" + str(l)] -= learn_rate * W_adjust
        params["b" + str(l)] -= learn_rate * b_adjust
    return params

def predict(X, Y, params, last="sigmoid"):
    A, cache = forward_prop(X, params, last=last)
    if last == "sigmoid":
        return (A > 0.5).astype(int)
    elif last == "relu":
        return A

def model(X, Y, dims, num_epochs=15000,
          learn_rate=0.00001, method="none", lambd=0,
          last="sigmoid", verbose=True, beta_1=0.9, beta_2=0.999,
          mini_batch_size=64):
    params = initialize(dims, "he")
    grads = {}

    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    for l in range(1, len(dims)):
        grads["VdW" + str(l)] = 0.001
        grads["Vdb" + str(l)] = 0.001
        grads["SdW" + str(l)] = 0.001
        grads["Sdb" + str(l)] = 0.001
    costs = []
    for i in range(num_epochs):
        for t, batch in enumerate(mini_batches):
            mini_batch_X, mini_batch_Y = batch
            A, caches = forward_prop(mini_batch_X, params, last=last)
            cost = compute_cost(mini_batch_Y, A,
                                params=params, method=method,
                                lambd=lambd, last=last)
            grads = back_prop(A, caches, mini_batch_Y, method=method, lambd=lambd,
                              last=last, beta_1=beta_1, beta_2=beta_2,grads_old=grads,
                              iter_num=t+1)
            params = optimize(params, grads, learn_rate)

        if (i+1) % 10 == 0:
            costs.append(cost)
        
    return params, costs

def count_boomerangs_validate(lst):
    return lst[0]==lst[-1]!=lst[1]


data_raw = [str(i) for i in range(1000)]
data_raw = ["0"*(3-len(i)) + i for i in data_raw]
data_raw = [list(i) for i in data_raw]
data_raw = [[int(i) for i in list] for list in data_raw]
data = np.array(data_raw).T

x_train = data / 10
x_test = data / 10

y_train = np.array([
    count_boomerangs_validate(x_train.T[i]) for i in range(x_train.shape[1])
    ]).reshape((1, x_train.shape[1]))
y_test = np.array([
    count_boomerangs_validate(x_test.T[i]) for i in range(x_test.shape[1])
    ]).reshape((1, x_test.shape[1]))

dims = [3, 8, 1]

np.random.seed(0)

params, costs = model(X = x_train,
                      Y = y_train,
                      dims = dims,
                      num_epochs = 1000,
                      learn_rate = 0.08,
                      method = "none",
                      lambd = .001,
                      last = "sigmoid",
                      verbose = True,
                      beta_1 = 0.9,
                      beta_2 = 0.999,
                      mini_batch_size = 64)

train_predictions = predict(x_train, 1, params)
test_predictions = predict(x_test, 1, params)

train_correct = (train_predictions == y_train)
test_correct = (test_predictions == y_test)

train_acc = (np.sum(train_correct) / x_train.shape[1]) * 100
test_acc = (np.sum(test_correct) / x_test.shape[1]) * 100

print("Training data accuracy: " + str(train_acc) + "%")
print("Testing data accuracy : " +str(test_acc) + "%")

def count_boomerangs(lst):
    global params
    lst = (np.array(lst) / 10).reshape(1, len(lst))
    potential_booms = []
    for i in range(2, len(lst[0])):
        potential_booms.append(lst[0][i-2:i+1].reshape(3, 1))
    booms = sum(predict(potential_boom, 1, params) for potential_boom in potential_booms)
    return int(booms)
