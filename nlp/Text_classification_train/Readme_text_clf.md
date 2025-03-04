

#### 基于Bert类对文本进行分类

##### 训练数据格式

     1. a.数据数据格式为Excel,包含3个字段
        text[str]，
        text_answer[str],
        flag[optional,int]:0表示负样本，1表示正样本，3表示重点正样本(提高重点对象出现频率)
        b.文本标签转数字标签映射字典：type_label2id_0207.json
     
     2. 执行训练任务命令：
        nohup python Query_clf_train.py  \
              -e 5 \
              -b 30 \
              -d data/Doctor_query_type_clf_train_data_norm_v0_240125.xlsx \ 
              -l data/doc_type_label2id_240125.json \
              -o doc_query_clf_model > log/doc_query_clf_log_240105.txt 2>&1 & 
        tail -F log/doc_query_clf_log_240105.txt














