# ...............DES.............
import random
import numpy as np
from Crypto.Cipher import DES
import matplotlib.pyplot as plt
from Crypto.Cipher import DES3
from Crypto.Random import get_random_bytes
import pandas as pd
import statistics
aval=[]
change_bits=[]
location=[]
data=[]
print(".....DES.....")
key = 88888888
des = DES.new(key.to_bytes(8, 'little'), DES.MODE_ECB)

p_text=0
print("Initial plain text generated:",bin(p_text))

ciphertext1 = des.encrypt(p_text.to_bytes(8, 'little'))
print("Entering Loop")


print("---------------------------------------------------------------------------")
p_text=p_text+1
n=p_text
dd=1
N=64
count=0
bit_count=0
for j in range(1,65):
    for i in range(1,65):
        count=count+1
        ra = n << dd
        m = (1 << N)
        b = ra % m
        c = n >> (N - dd)
        p_text = b | c
        print("Random bit stream generated:",bin(p_text))
        ciphertext2 = des.encrypt(p_text.to_bytes(8, 'little'))
        no_bits_changed=bin(p_text).count("1")-bit_count
        change_bits.append(no_bits_changed)
        print("No. of bits changed in plain the text :",no_bits_changed )
        xor_txt=int.from_bytes(ciphertext1, 'little')^int.from_bytes(ciphertext2, 'little')
        bin_array=bin(xor_txt)
        d = bin_array.count("1")
        print("Avalanche effect :",d)
        aval.append(d)
        data.append([no_bits_changed,d])
        counter=0
        location.clear()

        for p in range(0, len(bin_array)):
            if bin_array[p]=='1':
                location.append(p-2)#print(i, end=" ")
                counter=counter+1
        if j==64 and i==1:
            print("Final count:",count)
            break
        if i==64:
            ciphertext1=ciphertext2
            decrypted_text=des.decrypt(ciphertext2)
            iptext=bin(int.from_bytes(decrypted_text, 'little'))
            bit_count=iptext.count("1")
            print("Decrypted Text :",iptext)
        print("Locations of Avalanche Effect :",location)

        print("\n----------------")
        n=p_text
        print("Loop count................................................................................",count)
    p_text=p_text<<1
    p_text=p_text+1
    n=p_text
print("Final count",count)

# ...............3DES.............

print("------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------")
print("...........3DES..........")
des3_aval=[]
des3_change_bits=[]
des3_location=[]
des3_data=[]
while True:
    try:
        key = DES3.adjust_key_parity(get_random_bytes(24))
        break
    except ValueError:
        pass
cipher = DES3.new(key, DES3.MODE_ECB)


r_text=random.getrandbits(64)
print("Random bit stream generated:",bin(r_text))
print("----------------")
p_text=r_text^r_text
print("Random bit stream generated:",bin(p_text))


ciphertext1 = cipher.encrypt(p_text.to_bytes(8, 'little'))
p_text=0
print("---------------------------------------------------------------------------")
p_text=p_text+1
n=p_text
dd=1
N=64
count=0
bit_count=0
for j in range(1,65):
    for i in range(1,65):
        count=count+1
        ra = n << dd
        m = (1 << N)
        b = ra % m
        c = n >> (N - dd)
        p_text = b | c
        print("Random bit stream generated:",bin(p_text))
        ciphertext2 = msg = cipher.encrypt(p_text.to_bytes(8, 'little'))
        no_bits_changed=bin(p_text).count("1")-bit_count

        des3_change_bits.append(no_bits_changed)
        print("No. of bits changed in plain the text :",no_bits_changed )



        xor_txt=int.from_bytes(ciphertext1, 'little')^int.from_bytes(ciphertext2, 'little')
        bin_array=bin(xor_txt)
        d = bin_array.count("1")
        print("Avalanche effect :",d)
        des3_aval.append(d)
        des3_data.append([no_bits_changed,d])

        counter=0
        des3_location.clear()
        for p in range(2, len(bin_array)):
            if bin_array[p]=='1':
                des3_location.append(p-2)#print(i, end=" ")
                counter=counter+1
        if j==64 and i==1:
            print("Final count:",count)
            break
        if i==64:
            ciphertext1=ciphertext2
            cipher_decrypt = DES3.new(key, DES3.MODE_ECB)
            decrypted_text=cipher_decrypt.decrypt(ciphertext2)



            iptext=bin(int.from_bytes(decrypted_text, 'little'))
            bit_count=iptext.count("1")
            print("Decrypted Text :",iptext)
        n=p_text
    print("Loop count................................................................................",count)
    p_text=p_text<<1
    p_text=p_text+1
    n=p_text
print("Final count",count)


#to excel


df = pd.DataFrame(data, columns=['No of Bits Changed in the Plain Text - DES ','No of bits changed in the Cipher Text - DES'])
df2 = pd.DataFrame(des3_data, columns=['No of Bits Changed in the Plain Text - 3DES ','No of bits changed in the Cipher Text - 3DES'])
with pd.ExcelWriter('ONE_TEXT.xlsx') as writer:
    df.to_excel(writer, sheet_name='Avalanche(Text) DES')
    df2.to_excel(writer, sheet_name='Avalanche(Text) 3DES')

#to excel
print("\n----------------")


# Statistical Analysis

# Statistical Analysis

print("..................Statistical Analysis.............")
print("On Changing Plain Text Bits - DES")
print("Minimum : ", min(aval))
print("Maximum : ", max(aval))
print("Mean of DES : ", round(np.mean(aval), 2))
print("Median of DES : ", round(np.median(aval), 2))
print("Mode of DES : ", statistics.mode(aval))
print("Standard deviation :", np.std(aval))
print("On Changing Plain Text Bits - 3DES")
print("Minimum : ", min(des3_aval))
print("Maximum : ", max(des3_aval))
print("Mean of 3 DES : ", round(np.mean(des3_aval), 2))
print("Median of DES : ", round(np.median(des3_aval), 2))
print("Mode of DES : ", statistics.mode(des3_aval))
print("Standard deviation :", np.std(des3_aval))


# Graphs

fig = plt.figure("Avalanche Effect in DES/3DES  on changing Plain Text bits")

X_axis = np.arange(len(des3_change_bits))
#for i in range(0,64):
    #aval.append(0)
plt.bar(X_axis - 0.2, des3_aval, 0.4, label = '3 DES')
plt.bar(X_axis + 0.2, aval, 0.4,label = 'DES')


plt.xlabel("No. of bits changed in the Plain text")
plt.ylabel("Avalanche Effect (Bits flipped in the cipher text)")
plt.title("Avalanche Effect DES Vs 3DES on changing Plain Text bits")

plt.legend(['3 DES', 'DES'], loc='upper left')
plt.tight_layout()
plt.show()




fig = plt.figure("Avalanche Effect in DES  on changing Plain Text bits")

plt.hist(aval, bins=range(0,64,1), align='right',  edgecolor='black')
plt.title('Avalanche effect in DES')
plt.xlabel('No. of bits flipped in Cipher text')
plt.ylabel('Frequency of avalanche')
plt.show()


fig = plt.figure("Avalanche Effect in 3DES  on changing Plain Text bits")
plt.hist(des3_aval, bins=range(0,64,1), align='right',color="orange", edgecolor='black')
plt.title('Avalanche effect 3DES')
plt.xlabel('No. of bits flipped in Cipher text')
plt.ylabel('Frequency of avalanche')
plt.show()


import seaborn as sb
import matplotlib.pyplot as plt

plt.figure("Avalanche Effect in DES/3DES  on changing Plain Text bits")
sb.kdeplot(aval , bw = 0.5 , fill = False,multiple="stack")
sb.kdeplot(des3_aval , bw = 0.5 , fill = False,multiple="stack")
plt.legend(['DES','3 DES'],loc='upper left')
plt.xlabel(" No. of bits flipped in Cipher text")
sb.set_style("whitegrid")
plt.title("Avalanche Effect in DES/3DES - Density Plot")
plt.show()




