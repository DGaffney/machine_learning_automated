def univariate_test(x, y, model):
    scores = []
    x = np.matrix(x)
    for i in range(x.shape[1]):
         score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                  cv=ShuffleSplit(len(X), 3, .3))
         scores.append((round(np.mean(score), 3), names[i]))
    print sorted(scores, reverse=True)