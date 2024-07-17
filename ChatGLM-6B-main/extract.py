import time

import tqdm
from hanlp_restful import HanLPClient
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from nltk.corpus import wordnet as wn




# 文件命名地址
name_path = r"C:\Users\USER\Desktop\Test1\doc2\input_Name.txt"
text_path = r"C:\Users\USER\Desktop\Test1\doc2\input_text.txt"
nameText_path = r"C:\Users\USER\Desktop\Test1\doc2\nameText.txt"
nameExtract_path = r"C:\Users\USER\Desktop\Test1\doc2\nameExtract.txt"



# ##1. 首先从小说原文中找出所有已经得到的人名出现的句子，并且归纳到nameText文件中
# # 发现如果句子长度长于70行，那么就会爆显存，所以我需要使用hanlp的api进行句子的抽取式摘要提炼，生成不行
# HanLP = HanLPClient('https://www.hanlp.com/api', auth="NTk4MEBiYnMuaGFubHAuY29tOlRBeGJKb0V2VkEzcDk2OXI=", language='zh')
#
# # 读取名字列表
# with open(name_path, "r", encoding="gbk") as name_file:
#     names = [name.strip() for name in name_file.readlines()]
#
# # 读取文本内容
# with open(text_path, "r", encoding="gbk") as text_file:
#     lines = text_file.readlines()
#
# # 创建一个字典，用于存储每个名字对应的句子列表
# name_sentences = {name: [] for name in names}
#
# # 使用 tqdm 进行句子遍历并显示进度条
# for line in tqdm.tqdm(lines, desc="Processing sentences"):
#     for name in names:
#         if name in line and len(line) > 5:
#             name_sentences[name].append(line.strip())
#
# # 打开文件进行写入
# with open(nameText_path, "w", encoding="utf-8") as output_file:
#     for name, sentences in name_sentences.items():
#         summarized_sentences = []
#         if len(sentences) > 70:
#             chunk_size = 10 # 将句子分块，每块包含5个句子
#             for i in tqdm.tqdm(range(0, len(sentences), chunk_size), desc=f"Summarizing {name}"):
#                 chunk = sentences[i:i + chunk_size]
#                 while True:
#                     try:
#                         text = " ".join(chunk)
#                         summary = HanLP.extractive_summarization(text, topk=3)
#                         # 提取权重最大的摘要句子
#                         best_summary_sentence = max(summary.items(), key=lambda x: x[1])[0]
#                         summarized_sentences.append(best_summary_sentence)
#                         break  # 如果成功获取摘要，退出重试循环
#                     except Exception as e:
#                         if "HTTP Error 429" in str(e):
#                             print("请求过多，等待并重试...")
#                             time.sleep(60)  # 等待60秒
#                         else:
#                             raise e  # 抛出其他异常
#             # 将摘要结果写入文件
#             output_file.write(f"{name}\n")
#             output_file.write(" ".join(summarized_sentences) + "\n\n")
#         else:
#             # 如果句子数小于等于70，则直接写入文件
#             output_file.write(f"{name}\n")
#             output_file.write("\n".join(sentences) + "\n\n")
#
# print("找到了所有人名出现的句子，结果保存在", nameText_path)







# ##2. 使用ChatGLM对每个角色生成5个维度的特征，保存在nameExtraction文件中
# tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\USER\Desktop\ChatGLM-6B-main\THUDM\chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained(r"C:\Users\USER\Desktop\ChatGLM-6B-main\THUDM\chatglm-6b", trust_remote_code=True).half().cuda()
# model = model.eval()
#
# with open(nameText_path, "r", encoding="utf-8") as nameText:
#     lines = nameText.readlines()
#
# name_sentences = {}  # 存储每个角色的拼接句子
# current_name = None  # 当前处理的角色名
# current_text = []  # 当前角色的拼接句子
#
# # 使用 tqdm 显示处理进度
# for line in tqdm.tqdm(lines, desc="Processing characters"):
#     line = line.strip()
#     if line:
#         # 如果不是空行，说明是角色名或者句子
#         if current_name is None:
#             # 第一行是角色名
#             current_name = line
#         else:
#             # 后续行是角色的句子，将句子加入当前文本列表
#             current_text.append(line)
#     else:
#         # 空行表示当前角色的句子结束
#         if current_name and current_text:
#             # 拼接当前角色的句子为一行文本
#             name_sentences[current_name] = " ".join(current_text)
#             # 重置当前角色和文本列表
#             current_name = None
#             current_text = []
#
# # 如果最后一个角色没有空行结束，需要手动存储
# if current_name and current_text:
#     name_sentences[current_name] = " ".join(current_text)
#
# # 将结果保存到文件
# with open(nameExtract_path, "w", encoding="utf-8") as output_file:
#     for name, extracted_text in tqdm.tqdm(name_sentences.items(), desc="Generating features"):
#         output_file.write(f"{name}\n")
#         response, history = model.chat(tokenizer,
#                                        "请你帮助我根据以下的文本，总结文本中指定人物的五个维度：特征、习惯、目标、经验、关系，每个维度用一句话总结。对话的人物是" + name +"文本是："+ extracted_text,
#                                        history=[])
#         output_file.write(f"{response}\n\n")
#
# print("角色特征提取完成，结果保存在", nameExtract_path)








##3. 使用ATOMIC2020数据集构建一个普适性的知识图谱，然后仿照PEACOK论文对知识图谱进行预处理
## 之后我的预想是通过对ChatGLM总结出来的5个维度的人物特征为每个人物建立一个节点，然后将每个人物特征抽象成ATOMIC2020数据集中的最相近的某种人物特征。然后将二者相链接，这样我就可以表示出类似论文的知识图谱结构
data = pd.read_csv(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\train.tsv", sep='\t', header=None, names=['head', 'relation', 'tail'])

#1) 首先筛选出具有 "CapableOf" 关系的实体
capable_of_data = data[data['relation'] == 'CapableOf']
capable_of_data.to_csv(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\capable_of_entities.tsv", sep='\t', index=False)


#2) 筛选掉所有的动物的集合
# 获取所有动物的同义词集合
animal_synsets = wn.synsets('animal')
# 提取同义词集合中的动物名称
animal_names = set()
for synset in animal_synsets:
    animal_names.update(synset.lemma_names())
# 将动物名称写入文件
with open(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\animal_appendix.txt", 'w', encoding='utf-8') as file:
    for animal in animal_names:
        file.write(animal + '\n')

# 读取动物实体列表
with open('animal_appendix.txt', 'r') as file:
    animal_entities = [line.strip() for line in file]
# 移除动物实体
filtered_entities = capable_of_data[~capable_of_data['head'].isin(animal_entities)]
# 保存过滤后的数据
filtered_entities.to_csv(r"C:\Users\USER\Desktop\ChatGLM-6B-main\ATOMIC20\filtered_capable_of_entities.tsv", sep='\t', index=False)
















