# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\USER\Desktop\ChatGLM-6B-main\THUDM\chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(r"C:\Users\USER\Desktop\ChatGLM-6B-main\THUDM\chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, "请你帮助我根据以下的对话，总结对话中人物的五个维度：特征、习惯、目标、经验和关系。" + "人物是杨洋。杨洋是一位才华横溢的年轻女孩，从小就展现出非凡的音乐天赋。她的声音如同天籁，唱起歌来总是能够打动人心。杨洋不仅擅长唱歌，还喜欢自己创作歌曲。每当夜幕降临，安静的房间里总会传出她轻柔的歌声和吉他的伴奏，她的笔记本里写满了灵感和旋律。这些歌曲承载了她的梦想和希望，也是她与世界沟通的桥梁。杨洋的目标是拿到格莱美奖，她深知这个目标的艰难和挑战，但她从不退缩。在大学里，杨洋系统地学习了音乐理论和表演技巧，不断提高自己的专业素养。她总是积极参加各种音乐比赛和演出，不断积累舞台经验。大学时期的她，常常在校园的音乐厅里排练到深夜，精益求精地打磨每一个音符和节拍。尽管杨洋还在追逐梦想的道路上，但她已经拥有了大批忠实的粉丝。她的音乐通过网络传遍四方，感动了无数听众。粉丝们喜欢她的真诚和努力，每一首歌都能引起他们的共鸣。杨洋也深知粉丝的重要性，她常常在社交媒体上与他们互动，分享自己的创作历程和生活点滴。杨洋相信，只要坚持梦想，总有一天她会站在格莱美的舞台上，捧起那个象征着音乐最高荣誉的奖杯。而在此之前，她会继续唱下去，写下去，用音乐讲述她的故事，触动更多人的心灵。", history=[])
print(response)
