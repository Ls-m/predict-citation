from lr import r0,m0 
from runLSTM import r1,m1,r2,m2,r3,m3
import matplotlib.pyplot as plt
import seaborn as sns


rs = [r3,r2,r0,r1]
names = ['last','avg','lr','lstm']
print("r2 are ",rs)

data1 = {'Name':names, 'r2': rs} 
ax = sns.barplot(y= "r2",width = 0.3, x = "Name", data = data1, palette=("Blues_d"))
sns.set_context("poster")
plt.show()


