import os

for i in range(5):
    datadir=f'/local/s2071932/general_data/cbp_gold_small_v2/{i}/'
    os.system('export DATA_DIR='+datadir)
    n=os.system('./run_ner.sh')
    m=n>>8
    if m!=0:
        continue

