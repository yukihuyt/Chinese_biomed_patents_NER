import os
import numpy as np
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
print (str(np.argmax(memory_gpu)))