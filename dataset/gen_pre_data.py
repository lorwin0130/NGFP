import sys

dude_info_filename = 'dud-e info.txt'
dude_info_lst = []
with open(dude_info_filename,'r',encoding='utf-8') as f:
    dude_info_lst=f.readlines()
dude_info_lst = [term.strip().split('\t')[1:3] for term in dude_info_lst]

pd_lst=[]
for target_name,pdb_id in dude_info_lst:
    actives_filename = 'dud-e/{}/actives_final.ism'.format(target_name.lower())
    decoys_filename = 'dud-e/{}/decoys_final.ism'.format(target_name.lower())
    actives_lst, decoys_lst = [],[]
    with open(actives_filename,'r',encoding='utf-8') as f:
        actives_lst = f.readlines()
    with open(decoys_filename,'r',encoding='utf-8') as f:
        decoys_lst = f.readlines()
    for active in actives_lst:
        if active.strip()=='':
            continue
        pd_lst.append('{}\t{}\t{}\t{}\n'.format(target_name,pdb_id,active.split(' ')[0],'1'))
    for decoy in decoys_lst:
        if decoy.strip()=='':
            continue
        pd_lst.append('{}\t{}\t{}\t{}\n'.format(target_name,pdb_id,decoy.split(' ')[0],'0'))
    # break

# with open('pre_data.txt','w',encoding='utf-8') as f:
#     f.writelines(pd_lst)
with open('pre_data_all.txt','w',encoding='utf-8') as f:
    f.writelines(pd_lst)