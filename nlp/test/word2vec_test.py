import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec


# Refer to "Introduction to Deep Learning Using Python"
data = codecs.open("a_man_who_escaped_from_the_wall.txt","r",encoding="utf-16")
a = BeautifulSoup(data,"html.parser")
b = a.select_one("text > body")
text = b.getText()

twitter = Twitter()
results = []
lines = text.split("\n")
for line in lines:
    malist = twitter.pos(line,norm=True,stem=True)
    r = []
    for word in malist:
        if not word[1] in ['Josa',"Eomi","Punctutation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    results.append(rl)

    print(rl)

file = 'data.txt'
with open(file, 'w',encoding='utf-8') as fp:
    fp.write("\n".join(results))

data = word2vec.LineSentence(file)
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)
model.save("word2vec.model")
print("ok")

