# import spacy
# from transformers import pipeline
# import thinc
#
# # 加载 spaCy 模型
# nlp = spacy.load("en_core_web_sm")
#
# # 定义文章和人物
# text = """
# John is a famous singer and songwriter. He has been known for his excellent vocal skills and ability to write touching lyrics.
# In addition to his music career, he is also a philanthropist, actively involved in various charity events.
# """
# person_name = "John"
#
# # 使用 Hugging Face 的 transformers 提取属性
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# entities = ner_pipeline(text)
#
# # 从提取的实体中获取人物的属性句子
# sentences = [sent.text for sent in nlp(text).sents if person_name in sent.text]
#
# print("Relevant Sentences:", sentences)
#
# # 使用预训练语言模型提取属性
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#
# labels = ["singing", "songwriting", "philanthropy", "charity", "vocal skills"]
#
# for sentence in sentences:
#     results = classifier(sentence, candidate_labels=labels)
#     print(f"Sentence: {sentence}")
#     print("Labels:", results["labels"])
#     print("Scores:", results["scores"])





import spacy

# 加载模型
nlp = spacy.load("en_core_web_sm")

# 处理文本
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# 输出标记、词性、命名实体等
for token in doc:
    print(token.text, token.pos_, token.dep_)

for ent in doc.ents:
    print(ent.text, ent.label_)
