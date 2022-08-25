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

def test_overlapper(tdf, query,background,maxl = 10,graph_prefix=None):
    '''
    This function performs statistical analysis of alternative reading frame ORFs that overlap a larger background gene in a different frame.
    This works by subsetting the output to mutations which are synonymous in the background frame, and therefore should have no impact
    from selection on the effects on the background frame. This reduces our power somewhat, but generally enough mutations are detected to draw conclusions.
    '''
    ngene = tdf[tdf.Gene == background].set_index(['node_id',"NT"])

    def check_overlapper(row):
        try:
            return ngene.loc[row.node_id,row.NT].Synonymous
        except:
            return False
    sdf = tdf[(tdf.Gene == query)]
    sdf['NgeneSyn'] = sdf.apply(check_overlapper,axis=1)
    sdf = sdf[sdf.NgeneSyn]

    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    lvc = sdf[(~sdf.Synonymous) & (~sdf.IsStop)].Leaves.value_counts()
    slvc = sdf[sdf.IsStop].Leaves.value_counts()
    # print(olvc,lvc,slvc)
    p1 = [lvc.get(i,0) for i in range(1,maxl+1)]
    p1.append(sum([lvc[i] for i in lvc.index if i > maxl]))
    p2 = [olvc.get(i,0)*sum(p1) for i in range(1,maxl+1)]
    p2.append(sum([olvc[i] for i in olvc.index if i > maxl])*sum(p1))
    p3 = [slvc.get(i,0) for i in range(1,maxl+1)]
    p3.append(sum([slvc[i] for i in slvc.index if i > maxl]))
    p4 = [olvc.get(i,0)*sum(p3) for i in range(1,maxl+1)]
    p4.append(sum([olvc[i] for i in olvc.index if i > maxl])*sum(p3))
    # print(p1,p2)
    nstat,nspv = chisquare(p1,p2)
    nsef = np.sqrt(nstat/sum(p1))
    # print("Chisquare NS:",nspv)
    # print("NS Effect Size:",nsef)
    # print(p3,p4)
    sstat,spv = chisquare(p3,p4)
    ssef = np.sqrt(sstat/sum(p3))
    # print("Chisquare Stop:",spv)
    # print("Stop Effect Size:",ssef)
    
    if graph_prefix != None:
        distros = pd.concat([sdf[~sdf.Synonymous].Leaves.value_counts(normalize=True)[:maxl].rename("Nonsynonymous"),
                                sdf[sdf.Synonymous].Leaves.value_counts(normalize=True)[:maxl].rename("Synonymous"),
                                sdf[sdf.IsStop].Leaves.value_counts(normalize=True)[:maxl].rename("Early Stop")],axis=1)
        distros['Leaf Count'] = distros.index
        vdf = pd.melt(distros,id_vars='Leaf Count').rename({"variable":"Type","value":"Probability"},axis=1)
        sns.barplot(x='Leaf Count',y='Probability',hue='Type',data=vdf)
        plt.title(query+": Probability of Onward Transmission")
        plt.savefig(graph_prefix+"_"+query+".png")
    return nspv,nsef,spv,ssef

def test_independent(tdf,query,maxl=10,graph_prefix=None):
    '''
    This function performs statistical analysis of the leaf count distribution of an independent single gene in the standard frame.
    '''
    sdf = tdf[(tdf.Gene == query)]

    olvc = tdf[tdf.Synonymous].Leaves.value_counts(normalize=True)
    lvc = sdf[(~sdf.Synonymous) & (~sdf.IsStop)].Leaves.value_counts()
    slvc = sdf[sdf.IsStop].Leaves.value_counts()
    # print(olvc,lvc,slvc)
    p1 = [lvc.get(i,0) for i in range(1,maxl+1)]
    p1.append(sum([lvc[i] for i in lvc.index if i > maxl]))
    p2 = [olvc.get(i,0)*sum(p1) for i in range(1,maxl+1)]
    p2.append(sum([olvc[i] for i in olvc.index if i > maxl])*sum(p1))
    p3 = [slvc.get(i,0) for i in range(1,maxl+1)]
    p3.append(sum([slvc[i] for i in slvc.index if i > maxl]))
    p4 = [olvc.get(i,0)*sum(p3) for i in range(1,maxl+1)]
    p4.append(sum([olvc[i] for i in olvc.index if i > maxl])*sum(p3))
    nstat,nspv = chisquare(p1,p2)
    nsef = np.sqrt(nstat/sum(p1))
    # print("Chisquare NS:",nspv)
    # print("NS Effect Size:",nsef)
    # print(p3,p4)
    sstat,spv = chisquare(p3,p4)
    ssef = np.sqrt(sstat/sum(p3))
    # print("Chisquare Stop:",spv)
    # print("Stop Effect Size:",ssef)
    if graph_prefix != None:
        distros = pd.concat([sdf[~sdf.Synonymous].Leaves.value_counts(normalize=True)[:maxl].rename("Nonsynonymous"),
                                sdf[sdf.Synonymous].Leaves.value_counts(normalize=True)[:maxl].rename("Synonymous"),
                                sdf[sdf.IsStop].Leaves.value_counts(normalize=True)[:maxl].rename("Early Stop")],axis=1)
        distros['Leaf Count'] = distros.index
        vdf = pd.melt(distros,id_vars='Leaf Count').rename({"variable":"Type","value":"Probability"},axis=1)
        sns.barplot(x='Leaf Count',y='Probability',hue='Type',data=vdf)
        plt.title(query+": Probability of Onward Transmission")
        plt.savefig(graph_prefix+"_"+query+".png")
    return nspv,nsef,spv,ssef

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--tree",help="Path to a MAT protocol buffer containing the global phylogeny to analyze.")
    parser.add_argument("-r","--translation",help="Path to a translation table output from matUtils summary matching the input MAT. Will be generated if it does not already exist.", default='translation.tsv')
    parser.add_argument("-p","--prefix",help="Prefix to use for plotting output. If unused, no plots are saved.",default=None)
    parser.add_argument("-o","--output",help="Name of the output table to save statistical results to. Default is selection.tsv",default='selection.tsv')
    return parser.parse_args()

def primary_pipeline(treefile, translationfile, prefix=None, output='selection.tsv'):
    t = bte.MATree(treefile)
    if not exists(translationfile):
        print("Translation file not found; performing translation")
        subprocess.check_call("matUtils summary -i {} -g SARS-CoV-2.gtf -f NC_045512v2.fa -t {}".format(treefile,translationfile))
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
    #                     print(n.id,m,om)
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
    #Finally, we mask the 2% most highly homoplasic sites, again due to the enrichment of sequencing and assembly errors at these sites
    ntvc = tdf.NT.value_counts()
    thresh = np.percentile(ntvc,98)
    homoplasy = [i for i in ntvc.index if ntvc[i] > thresh]
    tdf = tdf[~tdf.NT.isin(homoplasy)]
    tdf['IsStop'] = tdf.CC.apply(lambda x:(x.split(">")[1] in ['TGA','TAA','TAG']))
    #we can now proceed to compute our statistical outputs and plots.
    odf = {k:[] for k in ['Gene','Mpv',"Npv","Mef","Nef"]}
    print("Computing statistics for alternate frame genes.")
    for subgene in ['ORF9b','ORF9c']:
        nspv,nsef,spv,ssef = test_overlapper(tdf,subgene,'N',2,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
    for subgene in ['ORF3b','ORF3c','ORF3d']:
        nspv,nsef,spv,ssef = test_overlapper(tdf,subgene,'ORF3a',2,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
    print("Computing statistics for full gene CDSs.")
    for gene in ['ORF1ab','S','E','M','N','ORF6','ORF8','ORF10']:
        nspv,nsef,spv,ssef = test_independent(tdf,gene,2,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
    print("Computing statistics for independent nsps.")
    for nsp in ['nsp3','nsp12_2','nsp2','nsp14','nsp13','nsp4','nsp15','nsp1','nsp16','nsp6','nsp5','nsp8','nsp10','nsp9','nsp7','nsp11','nsp12_1']:
        nspv,nsef,spv,ssef = test_independent(tdf,nsp,2,prefix)
        odf['Gene'].append(subgene)
        odf['Mpv'].append(nspv)
        odf['Npv'].append(spv)
        odf['Mef'].append(nsef)
        odf['Nef'].append(ssef)
    odf = pd.DataFrame(odf)
    odf.to_csv(output,sep='\t')
    print("Complete.")

def main():
    args = argparser()
    primary_pipeline(args.tree, args.translation, args.prefix, args.output)

if __name__ == '__main__':
    main()