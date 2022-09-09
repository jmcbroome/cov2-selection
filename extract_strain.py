import argparse
import bte

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--tree",help="Path to a MAT protocol buffer containing the global phylogeny to analyze.",required=True)
    parser.add_argument("-a","--annotation",help="Path to an annotation to extract.",required=True)
    parser.add_argument("-f","--reference",help="Path to the reference for the input MAT protocol buffer. Required for -of output.",default=None)
    parser.add_argument("-of","--outreference",help="Indicate a path for an imputed reference haplotype fasta representing the extracted strain.")
    parser.add_argument("-ot","--outtree",help="Path to write the subtree representing the indicated input annotation root and its descendants.")
    parser.add_argument("-v","--verbose",action='store_true',help='Use to print status updates.')
    return parser.parse_args() 

def parse_reference(refpath):
    refstr = []
    with open(refpath) as inf:
        for entry in inf:
            if entry[0] != ">":
                refstr.append(entry.strip())
    return "".join(refstr)

def process_mutstr(mstr):
    loc = int(mstr[1:-1])
    alt = mstr[-1]
    return (loc,alt)

def impute_haplotype(refstr, mutd):
    update = list(refstr)
    for m in mutd:
        loc,alt = process_mutstr(m)
        update[loc] = alt
    return "".join(update)

def main():
    args = argparser()
    t = bte.MATree(args.tree)
    available_annotes = t.dump_annotations()
    if args.annotation not in available_annotes:
        raise KeyError("Indicated annotation not found on the tree!")
    base_node = available_annotes[args.annotation]
    if args.verbose:
        print("Annotation identified on node {}; extracting subtree".format(base_node))
    subt = t.get_clade(args.annotation)
    if args.outtree:
        if args.verbose:
            print("Saving subtree for annotation {}".format(args.annotation))
        subt.save_pb(args.outtree)
    if args.outreference:
        if args.reference == None:
            raise ValueError("Input reference fasta (-f) must be used with option -of!")
        if args.verbose:
            print("Imputing haplotype fasta.")
        mutd = t.get_haplotype(base_node)
        refgenome = parse_reference(args.reference)
        newref = impute_haplotype(refgenome, mutd)
        with open(args.outreference,"w+") as of:
            print(">"+args.annotation,file=of)
            print(newref,file=of)
        

if __name__ == '__main__':
    main()