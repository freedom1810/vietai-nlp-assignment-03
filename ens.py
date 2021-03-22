import pandas as pd 

list_ens = ['/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold3_best_auc.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold2_best_loss.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold4_epoch5.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold1_best_loss.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold0_epoch3_best_loss.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold1_best_auc.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fodl3_best_loss.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold0_epoch6.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold4_epoch3.csv',
            '/content/drive/MyDrive/kaggle/vietai/vietai-nlp-assignment-03/res_fold2_best_auc.csv'
            ]
list_df = []
for df_dir in list_ens:
    list_df.append(pd.read_csv(df_dir))

res_df = {'id':[], 'goal_info':[], 'match_info':[], 'match_result':[], 'substitution':[], 'penalty':[], 'card_info': []}

for i in range(len(list_df[0])):

    goal_info = 0
    match_info = 0
    match_result = 0
    substitution = 0
    card_info = 0

    for df in list_df:
        goal_info += df.iloc[i]['goal_info']
        match_info += df.iloc[i]['match_info']
        match_result += df.iloc[i]['match_result']
        substitution += df.iloc[i]['substitution']
        card_info += df.iloc[i]['card_info']

    goal_info /= len(list_df)
    match_info /= len(list_df)
    match_result /= len(list_df)
    substitution /= len(list_df)
    card_info /= len(list_df)

    res_df['id'].append(i)
    res_df['goal_info'].append(goal_info)
    res_df['match_info'].append(match_info)
    res_df['match_result'].append(match_result)
    res_df['substitution'].append(substitution)
    res_df['penalty'].append(0)
    res_df['card_info'].append(card_info)


pd.DataFrame(res_df).to_csv('res_ens_bo5.csv', index = False)