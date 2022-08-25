# cov2-selection
Scripts for analysis of purifying selection across SARS-CoV-2 genes using global phylogenies.

To reproduce, create the environment:

```
conda create -f env.yml
conda activate selection
```

Download the latest global phylogeny:

```
wget http://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/public-latest.all.masked.pb.gz
```

Apply the included script.

```
python3 gene_wide_selection.py -t public-latest.all.masked.pb.gz -p selection -o pvalues.tsv
```

