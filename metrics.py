import itertools

def eval_dist_metric(tested, correct):
    score = 0 
    n = 0
    for key in correct:
        n+=1
        if key in tested:
            score+=abs(correct[key] - tested[key])
    
    return score/n

def check_result(communities):
   
    if len(communities) != 3:
        return False


    comset1 = set(communities[0])
    comset2 = set(communities[1])
    comset3 = set(communities[2])

    coms = [comset1, comset2, comset3]
    
    for perm in itertools.permutations(coms):
        if perm[0] == set(range(5)):
            if perm[1] == set(range(5,10)):
                if perm[2] == set(range(10,15)):       
                    return True
    return False