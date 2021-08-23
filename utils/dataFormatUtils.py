import collections

def fromTriple2Set( triples ):
    pos_dict, neg_dict = collections.defaultdict( set ), collections.defaultdict( set )
    for u, i, r in triples:
        if r < 1:
            pos_dict[ u ].add( i )
        else:
            neg_dict[ u ].add( i )
    return pos_dict, neg_dict


