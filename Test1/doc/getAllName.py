import hanlp
from tqdm import tqdm
import Levenshtein
import networkx as nx
import matplotlib.pyplot as plt

import re
# 输入和输出文件路径
input_text_path = r"/doc/input.txt"
split_text_path = r"/doc/test1.txt"  #第一步处理完分割开的句子
cut_text_path = r"/doc/test2.txt"  #第二步处理完经过hanlp的tokennizer模型分句后的结果
People_name_path = r"/doc/test3.txt"
relation_path = r"/doc/test4.txt"



# 1. 读取全文内容，并且根据常见的。？！：；符号进行分割
# 然后对所有的句子进行一下标准化处理，如果每一行中没有出现汉字的话那么直接删除这一行，然后开头不要留空白，如果句子开头是数字.那么就是标题也去掉
with open(input_text_path, "r", encoding="gbk") as file:
    input_text = file.read()
pattern = re.compile(r'([^。！？；：… —]+[。！？；：… —])')
sentences = pattern.findall(input_text)

# 如果还有剩余的句子
if input_text[len(''.join(sentences)):] != '':
    sentences.append(input_text[len(''.join(sentences)):])

# 去除不包含汉字的句子，去除前后空白字符，以及去除标题
processed_sentences = []
for sentence in sentences:
    sentence = sentence.strip()  # 去除前后空白字符
    if re.search(r'[\u4e00-\u9fff]', sentence) and not re.match(r'^\d+\.', sentence):
        processed_sentences.append(sentence)

# 获取句子总数,用来在进度条中显示
total_sentences = len(sentences)

with open(split_text_path, 'w', encoding='gbk') as file:
    for sentence in tqdm(sentences, desc="处理进度", total=total_sentences):
        file.write(sentence.strip() + '\n')
print(f"句子已保存到 {split_text_path}")









# 2. 用hanlp对每一行的句子进行分割处理，保存结果到test2.txt中
model_path = r"/pku98_6m_conv_ngram_20200110_134736"
tokenizer = hanlp.load(model_path)

with open(split_text_path, "r", encoding="gbk") as file:
    lines = file.readlines()

# 用于存储处理后的句子
processed_lines = []

# 分割和清理每一行
for line in tqdm(lines, desc="处理进度"):
    line = line.strip()
    if line:
        sentences = tokenizer(line)

        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if re.search(r'[\u4e00-\u9fff]', sentence) and not re.match(r'^\d+\.', sentence):
                cleaned_sentences.append(sentence)

        if len(cleaned_sentences) >= 5:
            processed_line = ' '.join(cleaned_sentences)
            processed_lines.append(processed_line)

# 写入输出文件
with open(cut_text_path, 'w', encoding='gbk') as file:
    for line in processed_lines:
        file.write(line + '\n')
print(f"句子已保存到 {cut_text_path}")









# 3. 用hanlp对分词后的句子进行命名实体识别,取出所有人名字的专有名词
# 加载 HanLP 的命名实体识别模型
test_PeopleName_model_path = r"/msra_ner_electra_small_20220215_205503"
ner_model = hanlp.load(test_PeopleName_model_path)

# 获取句子总数
with open(cut_text_path, "r", encoding="gbk") as f:
    num_lines = sum(1 for line in f)

# 处理分词后的句子文件，显示进度条
people_names = set()  # 使用集合保存人名，避免重复

with open(cut_text_path, "r", encoding="gbk") as file:
    lines = file.readlines()
    all_tokens = []  # 存储所有句子的 tokens
    for line in lines:
        tokens = line.strip().split()
        all_tokens.append(tokens)

    # 对所有句子进行命名实体识别
    for tokens in tqdm(all_tokens, desc="Processing sentences", total=num_lines):
        entities = ner_model([tokens], tasks='ner*')[0]
        for entity in entities:
            if entity[1] == 'PERSON':  # 判断实体类型是否为人名
                people_names.add(entity[0])

print(people_names)

# 进行人名筛选,发现有时候会出现老张，老松这种没有意义的名字，以及X这种名字
# 清理不完整的人名
def clean_name(name):
    # 删除包含特定模式的无效名字
    if "老" in name or len(name) <= 1:
        return None
    return name

def filter_similar_names(names):
    filtered_names = set()
    sorted_names = sorted(names, key=len, reverse=True)

    for name in sorted_names:
        cleaned_name = clean_name(name)
        if cleaned_name is None:
            continue
        if not any(Levenshtein.distance(cleaned_name, existing_name) <= 2 for existing_name in filtered_names):
            filtered_names.add(cleaned_name)

    return filtered_names

cleaned_names = set(filter(None, (clean_name(name) for name in people_names)))
final_names = filter_similar_names(cleaned_names)

# 将人名保存到文件中
with open(People_name_path, "w", encoding="gbk") as outfile:
    for name in final_names:
        outfile.write(name + "\n")
print(f"人名已保存到文件：{People_name_path}")








# #4. 用hanlp对分词后的句子进行依存解析，找出我已经找到的所有命名实体的关系
# sem_model_path = r"C:\Users\USER\Desktop\Test1\semeval16_sdp_electra_small_20220719_171433"
# sem_model = hanlp.load(sem_model_path)

# # 加载人名实体
# people_names_path = r"C:\Users\USER\Desktop\Test1\doc\test3.txt"
# people_names = set()
# with open(people_names_path, "r", encoding="gbk") as f:
#     for line in f:
#         name = line.strip()
#         if name:
#             people_names.add(name)
#
# # 处理分词后的句子文件
# with open(cut_text_path, "r", encoding="gbk") as f, open(relation_path, "w", encoding="gbk") as outfile:
#     lines = f.readlines()
#     for line in tqdm(lines, desc="处理句子", total=len(lines)):
#         tokens = line.strip().split()
#         if tokens:
#             try:
#                 parsed = sem_model(tokens)
#                 print(parsed)
#
#                 # 处理语义依存分析结果
#                 for token in parsed:
#                     # 仅处理和人名实体有关的依存关系
#                     if token['form'] in people_names:
#                         for dep in token['deps']:
#                             dep_id, rel = dep
#                             if parsed[dep_id - 1]['form'] in people_names:
#                                 # 写入实体1 实体2 关系 到文件中
#                                 outfile.write(f"{token['form']} {parsed[dep_id - 1]['form']} {rel}\n")
#
#             except Exception as e:
#                 print(f"处理句子时出现错误: {e}")
#                 print(f"句子 tokens: {tokens}")
# print(f"关系网络已保存到文件：{relation_path}")



# # 构建空的有向图
# G = nx.DiGraph()
#
# # 处理分词后的句子文件
# with open(cut_text_path, "r", encoding="gbk") as f:
#     lines = f.readlines()
#     for line in tqdm(lines, desc="处理句子", total=len(lines)):
#         tokens = line.strip().split()
#         if tokens:
#             try:
#                 parsed = sem_model(tokens)
#                 print(parsed)
#
#                 # 添加节点到图中
#                 for token in parsed:
#                     G.add_node(token['form'])
#
#                     # 处理依存关系
#                     for dep_id, rel in token['deps']:
#                         head_token = parsed[dep_id - 1]
#                         G.add_edge(head_token['form'], token['form'], label=rel)
#
#             except Exception as e:
#                 print(f"处理句子时出现错误: {e}")
#                 print(f"句子 tokens: {tokens}")
#
# # 绘制图形
# pos = nx.spring_layout(G)  # 选择布局算法
# labels = nx.get_edge_attributes(G, 'label')
# nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
# nx.draw_networkx_edges(G, pos, edge_color='gray')
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体表示中文
# plt.title("Semantic Dependency Graph")
# plt.axis('off')
# plt.show()
























