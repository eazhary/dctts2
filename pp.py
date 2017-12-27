import os
from glob import glob
import tqdm
import re

def clean(inp):
	re_list = r"[^ ًٌٍَُِْاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ.]" # Arabic character set
	out = re.sub(re_list,"",inp)
	return out

def clean2(inp):
	re_list = r"[^ اإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ.]" # Arabic character set
	out = re.sub(re_list,"",inp)
	return out


files = glob("d:/arabic/**/*.txt",recursive=True)

for i,file in enumerate(files):
	_,fname = os.path.split(file)
	print("Procesing file %s"%file)
	fout = open('%d_%s' % (i,fname), 'w', encoding='utf-8')
	with open(file, encoding='cp1256') as f:
		for line in f:
			parts = re.split('[,،.\)]',line.strip())
			for part in parts:
				c = clean(part)
				if len(c):
					c2 = clean2(c)
					fout.write("%s|%d|%s|%d\n" % (c,len(c),c2,len(c2)))
	fout.close()

