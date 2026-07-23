import matplotlib.pyplot as plt
import seaborn as sns
import topic_nmf

W, H, words = topic_nmf.run(6)

# Each word's topic = the column (topic) with the largest weight in its W row.
word_topic = W.argmax(axis=1)

print(f"\n{len(words)} words assigned across {W.shape[1]} topics\n")
for topic in range(W.shape[1]):
    # words in this topic, strongest first
    members = [i for i in range(len(words)) if word_topic[i] == topic]
    members.sort(key=lambda i: W[i, topic], reverse=True)
    print(f"Topic {topic} ({len(members)} words):")
    print("  " + ", ".join(words[i] for i in members))
    print()


fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(W, robust=True, annot=False, fmt=".2f", cmap = "coolwarm", cbar = True, ax=axs[0])
axs[0].set_title("W")
sns.heatmap(H, robust=True, annot=False, fmt=".2f", cmap = "coolwarm", cbar = True, ax=axs[1])
axs[1].set_title("H")
plt.tight_layout()
plt.show()