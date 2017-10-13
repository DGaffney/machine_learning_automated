import time
import model_info
times = []
paired = []
scores.append(score)
for i,model in enumerate(model_list()):
    # try:
    print i
    print model_list_str()[i]
    start = int(round(time.time() * 1000.0))
    clf = model
    score = cross_val_score(clf, x, y, cv=10)
    end = int(round(time.time() * 1000.0))
    times.append((end-start)/1000.0)
    paired.append([model_list_str()[i], (end-start)/1000.0])
    # except:
    #     print "Can't run"

fast_models()