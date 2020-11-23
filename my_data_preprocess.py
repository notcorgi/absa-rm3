import pandas as pd
import numpy as np

path = r"C:\Users\lishubin\Desktop\data_untag_stage3-文雅.csv"
csv_data = pd.read_csv(path, usecols=[3,6], header=0,encoding='ansi')  # 读取csv文件，usecols为列索引号,header为列名所在行号
print("原始数据\n", csv_data.head(10))  # 显示前十行
data_list=[]
label_list=[]
csv_exp_list=csv_data['explicit_text']
csv_object=csv_data['object']
for i in range(csv_exp_list.__len__()):
    if csv_object[i]==1:
        if type(csv_exp_list[i])!=float and 'cop' in csv_exp_list[i]:
            data_list.append(csv_exp_list[i])
            label_list.append("cop")
    if csv_object[i]==2:
        if type(csv_exp_list[i])!=float and 'Floyd' in csv_exp_list[i]:
            data_list.append(csv_exp_list[i])
            label_list.append("Floyd")
    if csv_object[i]==3:
        # print(i)
        if type(csv_exp_list[i])!=float and 'racism' in csv_exp_list[i]:
            data_list.append(csv_exp_list[i])
            label_list.append("racism")
    if csv_object[i]==6:
        if type(csv_exp_list[i])!=float and 'Trump' in csv_exp_list[i]:
            data_list.append(csv_exp_list[i])
            label_list.append("Trump")
savefile=pd.DataFrame({'data':data_list,"label":label_list})
# savefile['label']=label_list
savefile.to_csv(r'C:\Users\lishubin\Desktop\final_data.csv', index=False, encoding="utf-8",header=False,mode='a')
