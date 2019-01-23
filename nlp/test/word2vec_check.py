from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc

#생성된 모델 결과 확인하기
model = word2vec.Word2Vec.load('word2vec.model')
results = model.most_similar(positive=["당신"])

word = []
data = []

for result in results:
    word.append(result[0])
    data.append(result[1])

#자신의 컴퓨터 환경에서 한글 폰트 아무거나 경로 작성하면 됩니다.
font_name = font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf").get_name()
rc('font', family=font_name)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ypos = np.arange(len(word))
rects = plt.barh(ypos,data, align='center', height=0.5)
plt.yticks(ypos, word)

for i, rect in enumerate(rects):
    ax.text(0.95 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, str(data[i]) + '%', ha='right', va='center')
plt.show()