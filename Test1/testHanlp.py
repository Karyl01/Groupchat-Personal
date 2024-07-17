import hanlp

# model_path = r"C:\Users\USER\Desktop\Test1\msra_ner_electra_small_20220215_205503"
# ner_model = hanlp.load(model_path)
# print(ner_model([["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"], ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]], tasks='ner*'))

# # 获取HanLP的模型列表
# models = hanlp.pretrained.ALL
# for model_name, model_path in models.items():
#     if "SEMEVAL16_ALL_ELECTRA_SMALL_ZH" in model_name:
#         print(model_path)

# model_path = r"C:\Users\USER\Desktop\Test1\msra_ner_electra_small_20220215_205503"
# ner_model = hanlp.load(model_path)
# print(ner_model([ ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]], tasks='ner*'))




# import hanlp
# sem_model_path = r"C:\Users\USER\Desktop\Test1\semeval16_sdp_electra_small_20220719_171433"
# sem_model = hanlp.load(sem_model_path)
# graph = sem_model(["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"])
# print(graph)

# import hanlp
#
# # 加载语义依存分析模型
# sem_model_path = r"C:\Users\USER\Desktop\Test1\semeval16_sdp_electra_small_20220719_171433"
# sem_model = hanlp.load(sem_model_path)
#
# # 示例句子
# sentence = "马钢是白沐霖的朋友。"
#
# # 进行分词和依存分析
# parsed = sem_model(sentence)
#
# # 输出依存分析结果（示例）
# for token in parsed:
#     # 输出每个词语的信息（调试用）
#     print(token)
#
# # 提取白沐霖、马钢和他们之间的关系
# for token in parsed:
#     if token['form'] == '白沐霖' or token['form'] == '马钢':
#         for dep in token['deps']:
#             dep_id, rel = dep
#             if parsed[dep_id - 1]['form'] == '白沐霖' or parsed[dep_id - 1]['form'] == '马钢':
#                 # 输出或保存关系三元组
#                 print(f"{token['form']} {parsed[dep_id - 1]['form']} {rel}")
#
#
#
#




from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://www.hanlp.com/api', auth='NTk4MEBiYnMuaGFubHAuY29tOlRBeGJKb0V2VkEzcDk2OXI=')  # auth需要申请
doc = HanLP.parse('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')






























