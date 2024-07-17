from tqdm import tqdm
from nltk.corpus import wordnet as wn
import pandas as pd

from extract import capable_of_data


#1. 清洗动物的同义词，先获得动物的同义词，从wordnet中得到
# 获取所有动物的同义词集合
# animal_synsets = wn.synsets('animal')
# # 提取同义词集合中的动物名称
# animal_names = set()
# for synset in animal_synsets:
#     animal_names.update(synset.lemma_names())
# # 将动物名称写入文件
# with open(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\animal_appendix.txt", 'w', encoding='utf-8') as file:
#     for animal in animal_names:
#         file.write(animal + '\n')



#2. 对数据集进行动物实体和属性的清洗
# 读取动物实体列表
with open(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\animal_appendix.txt", 'r', encoding='utf-8') as file:
    animal_entities = [line.strip() for line in file]
# 使用 tqdm 显示进度条，过滤数据
filtered_entities = []
total = len(capable_of_data)
for idx, row in tqdm(capable_of_data.iterrows(), total=total, desc="Filtering entities"):
    if row['head'] not in animal_entities:
        filtered_entities.append(row)
# 转换为DataFrame
filtered_entities_df = pd.DataFrame(filtered_entities)
# 保存过滤后的数据
filtered_entities_df.to_csv(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\filtered_capable_of_entities.tsv", sep='\t', index=False)
print("过滤完成，结果保存在 filtered_capable_of_entities.tsv 文件中。")
