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
    for raa, osdf in tdf[(tdf.Gene.isin(['S','E','M','N']))].groupby('RefAA'):
        for aaa in aas:
            sdf = osdf[osdf.AltAA == aaa]
            if sdf.shape[0] < 25:
                hmap[raa].append(np.nan)
            else:
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
    regv = linregress(x=tdata.NormGrantham,y=tdata.nseffect)
    if graph != None:
        sns.scatterplot(x='Grantham',y='nseffect',data=raadf.replace("*",np.nan).dropna())
        plt.savefig(graph + "_correlation.png")
    return regv

def grantham_pipeline(fulltranslate_file,plot=None):
    tdf = pd.read_csv(fulltranslate_file,sep='\t')
    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    for g, sdf in tdf.groupby("Gene"):
        if plot != None:
            graph = plot + "_" + g
        aadf = build_effect_matrix(sdf, olvc, graph)
        regv = do_grantham(aadf, graph)
        print(g, regv.rvalue, regv.pvalue)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--fulltranslate",help="Path to an annotated full translation output table (from gene_wide_selection.py -d).")
    parser.add_argument("-p","--plot",help='Set a prefix to use to graph correlation and effect score heatmaps.')
    return parser.parse_args()

def main():
    args = argparser()
    grantham_pipeline(args.fulltranslate,args.plot)

if __name__ == '__main__':
    main()