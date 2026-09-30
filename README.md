
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# 1. Load Data
iris = load_iris(as_frame=True)
df = iris.frame
# 2. Visualization
sns.pairplot(df, hue="target")
plt.show()
# 3. Train Test Split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# 4. Model Training
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# 5. Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
