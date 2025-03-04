
#### 症状疾病相似度优化

    症状相似度：具体症状：突出【身体部位】+【症状】如头痛，手臂发痒；
                宽泛症状：皮肤有点痒(不知具体位置)，身体不适，头部不适(疼，痒，还是晕？)
      具体症状，应该和具体症状做相似，不出现宽泛症状(正负样本)？。

##### update-231122

     1.数据集  : Med_general_norm_category_231121.xlsx
       模型地址 : simi_model/Medical_simi-_General_-epochs_3-2023-11-22 
       处理脚本：Involve_update_simi_train_data_1120.ipynb
       训练脚本：Text_simi_train.py, Text_Similarity_Sampler.py






