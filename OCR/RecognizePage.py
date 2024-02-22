import numpy as np
# Memanggil function dari file .py lainnya.
from SegmentPage import segment_into_lines
from SegmentLine import segment_into_words
from RecognizeWord import recognize_words

#Ubah image menjadi bentuk array per setiap line
line_array=segment_into_lines('10.jpg')

#list untuk menampung index serta list kata
index_indicator=[]
words_list=[]
len_line_arr = 0

#Segmentasi setiap baris menjadi kata dan ditampung sebagai array.
for idx,im in enumerate(line_array):
    indicator,w_array=segment_into_words(im,idx)
    
    for k in range(len(w_array)):
        
        index_indicator.append(indicator[k])
        words_list.append(w_array[k])
        
    len_line_arr+=1
    
words_list=np.array(words_list)

#Melakukan reconition pada image
recognize_words(index_indicator,words_list,len_line_arr)

