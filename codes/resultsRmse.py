from lr import r0,m0 
from runLSTM import r1,m1,r2,m2,r3,m3
import matplotlib.pyplot as plt
import seaborn as sns





ms = [m3,m2,m0,m1]
names = ['last','avg','lr','lstm']
print("rmses are ",ms)

data1 = {'Name':names, 'rmse': ms} 
ax = sns.barplot(y= "rmse",width = 0.3, x = "Name", data = data1, palette=("spring_r"))
sns.set_context("poster")
plt.show()

