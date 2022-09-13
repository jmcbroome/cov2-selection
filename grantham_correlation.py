from scipy.stats import linregress, chisquare
import pandas as pd
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def build_effect_matrix(tdf, olvc, graph = None):
    aas = [a for a in tdf.RefAA.value_counts().index if a != "*"]
    hmap = {k:[] for k in aas}
    # hmap_count = {k:[] for k in aas}
    for raa, osdf in tdf.groupby('RefAA'):
        for aaa in aas:
            #skip to/from stops.
            if raa == "*" or aaa == "*":
                continue
            sdf = osdf[osdf.AltAA == aaa]
            lvc = sdf.Leaves.value_counts()
            p1 = [lvc.get(1,0), sum([lvc[i] for i in lvc.index if i > 1])]
            p2 = [olvc.get(1,0)*sum(p1), sum([olvc[i] for i in olvc.index if i > 1])*sum(p1)]
            nstat,nspv = chisquare(p1,p2)
            neff = nstat/sum(p1)
            #record nonsignificant synonymous substitutions
            #otherwise, leave it as nan, to represent that relationshp being ambiguous
            if nspv < 0.05 or aaa == raa:
                hmap[raa].append(neff)
            else:
                hmap[raa].append(np.nan)
            # hmap_count[raa].append(sdf.shape[0])
    aadf = pd.DataFrame(hmap,index=aas)
    if graph != None:
        plt.figure(figsize=[12,8])
        sns.heatmap(aadf)
        plt.savefig(graph + "_transition.png")
    return aadf

def do_grantham(raadf,graph=None):
    gdf = pd.read_csv("grantham.tsv",sep='\t').set_index("FIRST")
    gran = gdf.to_dict()
    def get_grantham(row):
        return max([gran.get(row.reference,{}).get(row.alternative,-1), gran.get(row.alternative,{}).get(row.reference,-1), 0])
    raadf['Grantham'] = raadf.apply(get_grantham,axis=1)
    #ignore stop codon mutations, as the stop is not part of the grantham matrix
    tdata = raadf.replace("*",np.nan).dropna()
    try:
        regv = linregress(x=tdata.Grantham,y=tdata.nseffect)
        if graph != None:
            sns.scatterplot(x='Grantham',y='nseffect',data=raadf.replace("*",np.nan).dropna())
            plt.savefig(graph + "_correlation.png")
    except ValueError:
        print("Could not compute or graph correlation!")
        return None
    return regv

def grantham_pipeline(fulltranslate_file,output,plot=None,aaout=None):
    odf = {k:[] for k in ['Gene','Total','Rvalue','Pvalue']}
    tdf = pd.read_csv(fulltranslate_file,sep='\t')
    tdf['RefAA'] = tdf.AA.apply(lambda x:x.split(":")[1][0])
    tdf['AltAA'] = tdf.AA.apply(lambda x:x.split(":")[1][-1])
    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    aavs = []
    for g, sdf in tdf.groupby("Gene"):
        if plot != None:
            graph = plot + "_" + g
        aadf = build_effect_matrix(sdf, olvc, graph)
        #columns are the reference, row indeces are the alternatives
        raadf = aadf.melt(ignore_index=False).reset_index().rename({"index":"alternative","variable":"reference","value":"nseffect"},axis=1)
        raadf['Gene'] = g
        aavs.append(raadf)
        regv = do_grantham(raadf, graph)
        if regv != None:
            print(g, regv.rvalue, regv.pvalue)
            odf['Gene'].append(g)
            odf['Total'].append(sdf.shape[0])
            odf['Rvalue'].append(regv.rvalue)
            odf['Pvalue'].append(regv.pvalue)
        else:
            print(g, "not computable")
    pd.DataFrame(odf).to_csv(output,sep='\t',index=False)
    if aaout != None:
        pd.concat(aavs).to_csv(aaout,sep='\t',index=False)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--fulltranslate",help="Path to an annotated full translation output table (from gene_wide_selection.py -d).",required=True)
    parser.add_argument("-p","--plot",help='Set a prefix to use to graph correlation and effect score heatmaps.')
    parser.add_argument("-o","--output",help='Set a name for the output summary file.',default='grantham_correlation.tsv')
    parser.add_argument("-a","--aaout",help='Save a table containing gene-specific long-form table of amino acid transition effects.',default=None)
    return parser.parse_args()

def main():
    args = argparser()
    grantham_pipeline(args.fulltranslate,args.output,args.plot,args.aaout)

if __name__ == '__main__':
    main()