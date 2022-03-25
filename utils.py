def dataset_summary(X,y):
    train_X, test_X = X[0], X[1]
    train_y, test_y = y[0], y[1]
    
    print()
    print("---------Dataset Summary----------")
    print("Total training samples:", len(train_X))
    print("Total training samples corresponding to class 0:", len(train_y[train_y[:]==0]))
    print("Total training samples corresponding to class 1:", len(train_y[train_y[:]==1]))
    print()
    print("Total testing samples:", len(test_X))
    print("Total testing samples corresponding to class 0:", len(test_y[test_y[:]==0]))
    print("Total testing samples corresponding to class 1:", len(test_y[test_y[:]==1]))
    print("----------------------------------")
    print()