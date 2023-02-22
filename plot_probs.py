import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
import ipdb
st = ipdb.set_trace
import torch.nn.functional as F
# st()
final_probs = F.softmax(torch.randn(1,1000),-1).squeeze()
plt.figure(figsize=(7,7))
plt.bar(np.arange(1000),final_probs.numpy())
pred_class = np.argmax(final_probs.numpy())
# sns.barplot(np.arange(1000),final_probs.numpy()).set_title(f" predictied class; {pred_class}")
plt.title(f" predictied class; {pred_class}")
# sns.title(f" predictied class; {pred_class}")

plt.savefig('probs.png')

# st()
print('hello')