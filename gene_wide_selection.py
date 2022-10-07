import bte
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from os.path import exists
import subprocess
from scipy.stats import chisquare
from statsmodels.stats.multitest import fdrcorrection as fdrc

def make_graph(sdf, query, graph_prefix):
    distros = pd.concat([sdf[~sdf.Synonymous].Leaves.value_counts(normalize=True).sort_index()[:10].rename("Nonsynonymous"),
                            sdf[sdf.Synonymous].Leaves.value_counts(normalize=True).sort_index()[:10].rename("Synonymous"),
                            sdf[sdf.IsStop].Leaves.value_counts(normalize=True).sort_index()[:10].rename("Early Stop")],axis=1)
    distros['Leaf Count'] = distros.index
    vdf = pd.melt(distros,id_vars='Leaf Count').rename({"variable":"Type","value":"Probability"},axis=1)
    sns.barplot(x='Leaf Count',y='Probability',hue='Type',data=vdf)
    plt.title(query+": Probability of Onward Transmission")
    plt.savefig(graph_prefix+"_"+query+".png")
    plt.clf()

def collect_stats(lvc,olvc,slvc={},threshold=1):
    #ugly solution with explicit iteration over the index because many individual values are missing and .loc doesn't allow for handling missing indeces
    p1 = [sum([lvc[i] for i in lvc.index if i <= threshold]), sum([lvc[i] for i in lvc.index if i > threshold])]
    p2 = [sum([olvc[i] for i in olvc.index if i <= threshold])*sum(p1), sum([olvc[i] for i in olvc.index if i > threshold])*sum(p1)]
    nstat,nspv = chisquare(p1,p2)
    nsef = np.sqrt(nstat/sum(p1))
    if len(slvc) > 0:
        p3 = [sum([slvc[i] for i in slvc.index if i <= threshold]), sum([slvc[i] for i in slvc.index if i > threshold])]
        p4 = [sum([olvc[i] for i in olvc.index if i <= threshold])*sum(p3), sum([olvc[i] for i in olvc.index if i > threshold])*sum(p3)]
        sstat,spv = chisquare(p3,p4)
        ssef = np.sqrt(sstat/sum(p3))
    else:
        sstat,spv,ssef = np.nan,np.nan,np.nan
    return nspv,nsef,spv,ssef

def build_site_table(tdf,siteout,threshold=1):
    idf = {k:[] for k in ['Gene','Site','Count','NSpv','NSeffect','STpv','STeffect','SingleRate']}
    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    for key, sdf in tqdm(tdf.groupby(["Gene",'AAL'])):
        g,site = key
        lvc = sdf[(~sdf.Synonymous) & (~sdf.IsStop)].Leaves.value_counts()
        if len(lvc) == 0:
            continue
        slvc = sdf[(sdf.IsStop)].Leaves.value_counts()
        nspv,nsef,spv,ssef = collect_stats(lvc,olvc,slvc,threshold=threshold)
        idf['Gene'].append(g)
        idf['Site'].append(site)
        idf['Count'].append(sdf[(~sdf.Synonymous) & (~sdf.IsStop)].shape[0])
        idf['NSpv'].append(nspv)
        idf['NSeffect'].append(nsef)
        idf['STpv'].append(spv)
        idf['STeffect'].append(ssef)
        idf['SingleRate'].append(lvc.get(1,0))
    idf = pd.DataFrame(idf)
    idf['FDRnsPV'] = fdrc(idf.NSpv)[1]
    idf['SingleProp'] = (idf.SingleRate/idf.Count)
    idf.to_csv(siteout,sep='\t',index=False)

def test_overlapper(tdf,query,background,threshold=1,graph_prefix=None):
    '''
    This function performs statistical analysis of alternative reading frame ORFs that overlap a larger background gene in a different frame.
    This works by subsetting the output to mutations which are synonymous in the background frame, and therefore should have no impact
    from selection on the effects on the background frame. This reduces our power somewhat, but generally enough mutations are detected to draw conclusions.
    '''
    ngene = tdf[tdf.Gene == background].set_index(['node_id',"NT"])
    assert "Synonymous" in tdf.columns
    def check_overlapper(row):
        try:
            return ngene.loc[row.node_id,row.NT].Synonymous
        except:
            return False
    sdf = tdf[(tdf.Gene == query)]
    assert sdf.shape[0] > 0
    sdf['NgeneSyn'] = sdf.apply(check_overlapper,axis=1)
    sdf = sdf[sdf.NgeneSyn]
    assert "Synonymous" in sdf.columns

    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    lvc = sdf[(~sdf.Synonymous) & (~sdf.IsStop)].Leaves.value_counts()
    slvc = sdf[sdf.IsStop].Leaves.value_counts()
    nspv,nsef,spv,ssef = collect_stats(lvc,olvc,slvc,threshold=threshold)
    if graph_prefix != None:
        try:
            make_graph(sdf, query, graph_prefix)
        except:
            print("Unable to graph gene {}. Continuing".format(query))
    return nspv,nsef,spv,ssef,sdf.shape[0]

def test_independent(tdf,query,threshold=1,graph_prefix=None):
    '''
    This function performs statistical analysis of the leaf count distribution of an independent single gene in the standard frame.
    '''
    sdf = tdf[(tdf.Gene == query)]

    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    lvc = sdf[(~sdf.Synonymous) & (~sdf.IsStop)].Leaves.value_counts()
    slvc = sdf[sdf.IsStop].Leaves.value_counts()
    nspv,nsef,spv,ssef = collect_stats(lvc,olvc,slvc,threshold=threshold)
    if graph_prefix != None:
        try:
            make_graph(sdf, query, graph_prefix)
        except:
            print("Unable to graph gene {}. Continuing".format(query))
    return nspv,nsef,spv,ssef,sdf.shape[0]

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--tree",help="Path to a MAT protocol buffer containing the global phylogeny to analyze.",required=True)
    parser.add_argument("-r","--translation",help="Path to a translation table output from matUtils summary matching the input MAT. Will be generated if it does not already exist.", default='translation.tsv')
    parser.add_argument("-p","--prefix",help="Prefix to use for plotting output. If unused, no plots are saved.",default=None)
    parser.add_argument("-o","--output",help="Name of the output table to save statistical results to. Default is selection.tsv",default='selection.tsv')
    parser.add_argument("-s","--siteout",help="Set to a name to produce a site-specific analysis table.",default=None)
    parser.add_argument("-d","--data",help="Set to a name to save the processed translation table for use with additional scripts.",default=None)
    parser.add_argument("-c","--cutoff",help="Set the number of leaves at which to partition the data for analysis. Must be positive. Default 1",type=int,default=1)
    return parser.parse_args()

def primary_pipeline(treefile, translationfile, prefix=None, output='selection.tsv', siteout=None, data=None, cutoff=1):
    t = bte.MATree(treefile)
    if not exists(translationfile):
        print("Translation file not found; performing translation")
        subprocess.check_call("matUtils summary -i {} -g SARS-CoV-2.gtf -f NC_045512v2.fa -t {}".format(treefile,translationfile),shell=True)
    otdf = pd.read_csv(translationfile,sep='\t').set_index("node_id")
    ##First, we mask all mutations which are reversions of previous mutations
    ##as contamination often leads to spurious recurring or highly homoplasic reversions that cannot be transmitted to descendants
    print("Masking highly homoplasic mutations and problematic sites.")
    def do_masking(t,maximum=1):
        maskd = {}
        for n in tqdm(t.depth_first_expansion()):
            to_mask = set()
            mutations = n.mutations
            for ancestor in t.rsearch(n.id,True):
                for m in mutations:
                    om = m[-1] + m[1:-1] + m[0]
                    if om in ancestor.mutations:
                        to_mask.add(m)
            if len(to_mask) > maximum:
                maskd[n.id] = to_mask
        return maskd

    maskd = do_masking(t,0)    
    tdf = {k:[] for k in ['node_id','AA','NT','CC',"Leaves"]}
    for i,d in tqdm(otdf.dropna().iterrows()):
        nts = d.nt_mutations.split(";")
        ccs = d.codon_changes.split(";")
        for ai, aa in enumerate(d.aa_mutations.split(";")):
            nt = nts[ai]
            if nt in maskd.get(i,[]):
                continue
            tdf['node_id'].append(i)
            tdf['AA'].append(aa)
            tdf['NT'].append(nt)
            tdf['CC'].append(ccs[ai])
            tdf['Leaves'].append(d.leaves_sharing_mutations)
    tdf = pd.DataFrame(tdf)

    def get_aal(aa):
        aal = aa.split(":")[1]
        return int(aal[1:-1])
    tdf['AAL'] = tdf.AA.apply(get_aal)
    tdf['Gene'] = tdf.AA.apply(lambda x:x.split(":")[0])
    tdf['Loc'] = tdf.NT.apply(lambda x:int(x.split(",")[0][1:-1]))
    tdf['Synonymous'] = tdf.AA.apply(lambda x:(x.split(":")[1][0] == x.split(":")[1][-1]))
    ##We further mask known problematic sites with extremely high mutation rates or enrichment for sequencing errors
    ##https://virological.org/t/issues-with-sars-cov-2-sequencing-data/473
    sites_to_mask = [187, 1059, 2094, 3037, 3130, 6990, 8022, 10323, 10741, 11074, 13408, 14786, 19684, 20148, 21137, 24034, 24378, 25563, 26144, 26461, 26681, 28077, 28826, 28854, 29700]
    sites_to_mask.extend([4050, 13402, 11083, 15324, 21575])
    tdf = tdf[~tdf.Loc.isin(sites_to_mask)]
    if data != None:
        tdf.to_csv(data,sep='\t',index=False)
    # Disable this filter for now due to the apparent importance of many highly-mutated key binding sites
    # #Finally, we mask the 2% most highly homoplasic sites, again due to the enrichment of sequencing and assembly errors at these sites
    # ntvc = tdf.NT.value_counts()
    # thresh = np.percentile(ntvc,98)
    # homoplasy = [i for i in ntvc.index if ntvc[i] > thresh]
    # tdf = tdf[~tdf.NT.isin(homoplasy)]
    tdf['IsStop'] = tdf.CC.apply(lambda x:(x.split(">")[1] in ['TGA','TAA','TAG']))
    assert tdf.shape[0] > 0
    #we can now proceed to compute our statistical outputs and plots.
    odf = {k:[] for k in ['Gene','Mpv',"Npv","Mef","Nef","MutationCount"]}
    print("Computing statistics for alternate frame genes.")
    for subgene in ['ORF9b','ORF9c']:
        nspv,nsef,spv,ssef,mc = test_overlapper(tdf,subgene,'N',cutoff,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
        odf['MutationCount'].append(mc)
    for subgene in ['ORF3b','ORF3c','ORF3d']:
        nspv,nsef,spv,ssef,mc = test_overlapper(tdf,subgene,'ORF3a',cutoff,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
        odf['MutationCount'].append(mc)
    print("Computing statistics for full gene CDSs.")
    for gene in ['ORF1ab','S','E','M','N','ORF6','ORF8','ORF10']:
        nspv,nsef,spv,ssef,mc = test_independent(tdf,gene,cutoff,prefix)
        odf['Gene'].append(gene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
        odf['MutationCount'].append(mc)
    print("Computing statistics for independent nsps.")
    for nsp in ['nsp3','nsp12_2','nsp2','nsp14','nsp13','nsp4','nsp15','nsp1','nsp16','nsp6','nsp5','nsp8','nsp10','nsp9','nsp7','nsp11','nsp12_1']:
        nspv,nsef,spv,ssef,mc = test_independent(tdf,nsp,cutoff,prefix)
        odf['Gene'].append(nsp)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
        odf['MutationCount'].append(mc)
    odf = pd.DataFrame(odf)
    odf.to_csv(output,sep='\t',index=False)
    if siteout != None:
        build_site_table(tdf,siteout,threshold=cutoff)

    print("Complete.")

def main():
    args = argparser()
    if args.cutoff <= 0:
        raise ValueError("Cutoff value must be a positive integer!")
    primary_pipeline(args.tree, args.translation, args.prefix, args.output, args.siteout, args.data, args.cutoff)

if __name__ == '__main__':
    main()