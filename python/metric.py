

def l(p, G, weight="length"):
    l_p = 0
    for e in range(len(p)-1):
        l_p += 1 #G[p[e]][p[e+1]][0][weight]
    return l_p


def get_overlap(p_1, p, G, l_p=-1, debug=False, weight="length"):
    if(l_p < 0):
        l_p = l(p, G, weight)
    overlap = []
    for n in p_1:
        if(n in p):
            overlap.append(n)
    l_ov = 0
    
    for e in range(len(overlap)-1):
        if(overlap[e+1] in G[overlap[e]]):
            l_ov += 1 #G[overlap[e]][overlap[e+1]][0][weight]

    if(debug):
        print(len(p_1), len(p), len(overlap), l_ov, l_p)
    return l_ov/l_p
    
        