import random
from Crypto.Cipher import DES
from Crypto.Cipher import DES3
from Crypto import Random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
#des
aval=[]
change_bits=[]
location=[]
data=[]
print(".....DES.....")


p_text=b'ABCDEFGH' #plain text
key = 0
des = DES.new(key.to_bytes(8, 'little'), DES.MODE_ECB)
ciphertext1 = des.encrypt(p_text)
print("gg",key,ciphertext1)
print("ptext",des.decrypt(ciphertext1))



print("Entering Loop")
print("---------------------------------------------------------------------------")
key=key+1
n=key
dd=1
N=64
count=0
for j in range(1,65):
    for i in range(1,65):
        count=count+1
        ra = n << dd
        m = (1 << N)
        b = ra % m
        c = n >> (N - dd)
        key = b | c
        print("New Key Generated :",bin(key),"count",c)
        no_bits_changed=bin(key).count("1")
        change_bits.append(no_bits_changed)
        print("No of bits changed in key :",no_bits_changed)
        des = DES.new(key.to_bytes(8, 'little'), DES.MODE_ECB)
        ciphertext2 = des.encrypt(p_text)
        print("gg",key,ciphertext2)
        print("ptext",des.decrypt(ciphertext2))

        xor_txt=int.from_bytes(ciphertext1, 'little')^int.from_bytes(ciphertext2, 'little')
        bin_array=bin(xor_txt)
        d = bin_array.count("1")
        print("Avalanche effect : ",d)
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
        print("Locations of Avalanche Effect :",location)
        n=key
    print("Loop count................................................................................",count)
    key=key<<1
    key=key+1
    n=key
print("Final count",count)


# ...............3DES.............
des3_aval=[]
des3_change_bits=[]
des3_location=[]
des3_data=[]
key = random.getrandbits(192)
#print("Random key in binary (192bits) :",bin(key),"Length :", len(bin(key)))
key=key>>64
#print("Random key reduced to 64 bits :",bin(key),"Length :", len(bin(key)))
key_initial=key<<64
print("Key in binary (129 bits) :",bin(key_initial),"Length :", len(bin(key_initial)))

print("------------------")
plaintext = 'ABCDEFGH'
print("Plain Text :",plaintext)
plaintext_bytes = bytes(plaintext, 'utf-8')
plaintext_bits=bin(int.from_bytes(plaintext_bytes, 'little'))
print("Plain Text in binary :",plaintext_bits)

cipher_encrypt = DES3.new(key_initial.to_bytes(24), DES3.MODE_ECB)
encrypted_text1 = cipher_encrypt.encrypt(plaintext_bytes)

print("Encrpted Text :",encrypted_text1)
encrypted_text_bits1=bin(int.from_bytes(encrypted_text1))
print("Encrpted Text in binary:",encrypted_text_bits1)

cipher_decrypt = DES3.new(key_initial.to_bytes(24), DES3.MODE_ECB)
decrypted_text1=cipher_decrypt.decrypt(encrypted_text1)

print("Decrypted Text :",decrypted_text1)
print("------------------")
#------------

print("initial key",bin(key_initial),len(bin(key_initial)))


print("Entering Loop")
temp_key1=0
print("---------------------------------------------------------------------------")
temp_key1=temp_key1+1
n=temp_key1
dd=1
N=64
count=0
for j in range(1,65):
    for i in range(1,65):
        count=count+1
        ra = n << dd
        m = (1 << N)
        b = ra % m
        c = n >> (N - dd)
        temp_key1 = b | c
        temp_key=key_initial^temp_key1
        print("generated key",bin(temp_key),len(bin(temp_key)))
        #insert

        print("------------------")
        print("Plain Text :",plaintext)
        print("Plain Text in binary :",plaintext_bits)
        cipher_encrypt = DES3.new(temp_key.to_bytes(24), DES3.MODE_ECB)
        encrypted_text2 = cipher_encrypt.encrypt(plaintext_bytes)
        print("Encrpted Text :",encrypted_text1)

        encrypted_text_bits2=bin(int.from_bytes(encrypted_text2))
        print("Encrpted Text in binary:",encrypted_text_bits2)

        cipher_decrypt = DES3.new(temp_key.to_bytes(24), DES3.MODE_ECB)
        decrypted_text1=cipher_decrypt.decrypt(encrypted_text2)

        print("Decrypted Text :",decrypted_text1)
        kdiff=key_initial^temp_key
        bit_change_count=bin(kdiff).count("1")
        des3_change_bits.append(bit_change_count)
        ava_diff=bin(int.from_bytes(encrypted_text1)^int.from_bytes(encrypted_text2))
        aval_count=ava_diff.count("1")
        des3_aval.append(aval_count)
        des3_data.append([bit_change_count,aval_count])
        print("No of bits changed in key :",bit_change_count)
        print("Avalanche effect : ", aval_count)

        counter=0
        des3_location.clear()

        for l in range(0, len(ava_diff)):
            if ava_diff[l]=='1':
                des3_location.append(l-2)#print(i, end=" ")
                counter=counter+1
        if j==64 and i==1:
            print("Final count:",count)
            break
        print("Locations of Avalanche Effect :",des3_location)
        n=temp_key1
    print("Loop count................................................................................",count)
    temp_key1=temp_key1<<1
    temp_key1=temp_key1+1
    n=temp_key1
print("Final count",count)

df = pd.DataFrame(data, columns=['No of Bits Changed in the Key - DES ','No of bits changed in the Cipher Text - DES'])
df2 = pd.DataFrame(des3_data, columns=['No of Bits Changed in the Key - 3DES ','No of bits changed in the Cipher Text - 3DES'])
with pd.ExcelWriter('AVALANCHE_KEY.xlsx') as writer:
    df.to_excel(writer, sheet_name='Avalanche(Key) DES')
    df2.to_excel(writer, sheet_name='Avalanche(Key) 3DES')

#to excel
print("\n----------------")


# Statistical Analysis

print("..................Statistical Analysis.............")
print("On Changing Key Bits - DES")
print("Minimum : ", min(aval))
print("Maximum : ", max(aval))
print("Mean of DES : ", round(np.mean(aval), 2))
print("Median of DES : ", round(np.median(aval), 2))
print("Mode of DES : ", statistics.mode(aval))
print("Standard deviation :", np.std(aval))
print("On Changing Key Bits - 3DES")
print("Minimum : ", min(des3_aval))
print("Maximum : ", max(des3_aval))
print("Mean of 3 DES : ", round(np.mean(des3_aval), 2))
print("Median of DES : ", round(np.median(des3_aval), 2))
print("Mode of DES : ", statistics.mode(des3_aval))
print("Standard deviation :", np.std(des3_aval))



# Graphs

fig = plt.figure("Avalanche Effect in DES/3DES  on changing Key bits")

X_axis = np.arange(len(des3_change_bits))
#for i in range(0,64):
    #aval.append(0)
plt.bar(X_axis - 0.2, des3_aval, 0.4, label = '3 DES',color='#DAA03DFF')
plt.bar(X_axis + 0.2, aval, 0.4,label = 'DES',color='#616247FF')


plt.xlabel("No. of bits changed in the Key")
plt.ylabel("Avalanche Effect (Bits flipped in the cipher text)")
plt.title("Avalanche Effect DES Vs 3DES on changing Key bits")

plt.legend(['3 DES', 'DES'], loc='upper left')
plt.tight_layout()
plt.show()




fig = plt.figure("Avalanche Effect in DES  on changing Key bits")

plt.hist(aval, bins=range(0,64,1), align='right',  edgecolor='black',color='#DAA03DFF')
plt.title('Avalanche effect in DES')
plt.xlabel('No. of bits flipped in Cipher text')
plt.ylabel('Frequency of avalanche')
plt.show()


fig = plt.figure("Avalanche Effect in 3DES  on changing Key bits")
plt.hist(des3_aval, bins=range(0,64,1), align='right',color='#616247FF', edgecolor='black')
plt.title('Avalanche effect 3DES')
plt.xlabel('No. of bits flipped in Cipher text')
plt.ylabel('Frequency of avalanche')
plt.show()


import seaborn as sb
import matplotlib.pyplot as plt

plt.figure("Avalanche Effect in DES/3DES  on changing Key bits")
sb.kdeplot(aval , bw = 0.5, fill = False,multiple="stack",color='#DAA03DFF')
sb.kdeplot(des3_aval , bw = 0.5 , fill = False,multiple="stack",color='#616247FF')
plt.legend(['DES','3 DES'],loc='upper left')
plt.xlabel(" No. of bits flipped in Cipher text")
sb.set_style("whitegrid")
plt.title("Avalanche Effect in DES/3DES - Density Plot")
plt.show()





