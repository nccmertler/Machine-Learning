import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns
import plotly.express as px



#The code block below reads the data from a CSV file saved on my computer, assigns it to a variable named `data_copy`, and deletes the `customer_id` column from the data.

data=pd.read_csv("D:/Microsoft VS Code/Machine_Learning_Hw/WA_Fn-UseC_-Telco-Customer-Churn_dataset.csv")
data_copy=data.copy()
data_copy=data_copy.drop(columns="customerID", axis=1)

data_copy["SeniorCitizen"]=["Yes" if kod==1 else "No" for kod in data_copy["SeniorCitizen"]]
data_copy["Partner"]=["Married" if kod=="Yes" else "Single" for kod in data_copy["Partner"]]
data_copy["PhoneService"]=["Available" if kod==1 else "Not Available" for kod in data_copy["PhoneService"]]
data_copy["MultipleLines"]=["Available" if kod=="Yes" else "Not Available" for kod in data_copy["MultipleLines"]]
data_copy["OnlineSecurity"]=["Available" if kod=="Yes" else "Not Available" for kod in data_copy["OnlineSecurity"]]
data_copy["Churn"]=[1 if kod=='Yes' else 0 for kod in data_copy["Churn"]]

#Chect the data and datatype
print(data_copy.info())

#Change totalcharges object to float. 
data_copy["TotalCharges"]=pd.to_numeric(data_copy["TotalCharges"], errors="coerce")

#Check empty data is available.
print(data_copy.isnull().sum())

#Missing observations can be either deleted or filled with the mean. However, this method is not always applicable. 
#For instance, it wouldn't make sense to take an average value that includes both those who have a monthly TV subscription or security service subscription and those who don't.
#In such cases, it is necessary to establish a relationship between the data. It makes sense for the total charges to be part of a relationship like total charges = monthly charges * tenure

print(data_copy[data_copy["tenure"]==0])

#There are 11 records with tenure=0. Therefore, instead of imputing, it would be more logical to delete the empty records at this stage.
data_copy=data_copy.dropna()


#Lets try labelencoder for categorical variable
le=LabelEncoder()
categorical=data_copy.select_dtypes(include="object").columns
data_copy.update(data_copy[categorical].apply(le.fit_transform))

#Let's separate our data into inputs and outputs. The data we want to show as output, which we are curious about, is churn. Therefore, let's assign churn to the y variable
y=data_copy["Churn"]
X=data_copy.drop(columns="Churn", axis=1)

#Lets Divide into test and train data.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Let's normalize the features in the dataset to transform them into a distribution with a mean of 0 and a standard deviation of 1.
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

models=["Linear SVC", "SVC","Ridge", "Logistic", "RandomForest", "LGBM", "XGBM"]

#I performed Hyperparameter tuning tuning for each model type within the classifier to find the optimal values. Since it took a long time, I am indicating these codes in the comment section at the bottom.
classifier=[LinearSVC(random_state=0, C=0.1,penalty="l1", dual=False) ,SVC(random_state=0, C=1, gamma=0.01), RidgeClassifier(random_state=0, alpha=0.1),
          LogisticRegression(random_state=0,C=0.1,penalty="l2",dual=False), RandomForestClassifier(random_state=0, max_depth=10, min_samples_split=2, n_estimators=2000), 
          LGBMClassifier(random_state=0, learning_rate=0.01, max_depth=4, n_estimators=2000, subsample=0.6),XGBClassifier(learning_rate=0.01,n_estimators=2000,max_depth=4, subsample=0.8 )]


def education(model):
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
    else:
        probas = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = roc_auc_score(y_test, probas)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)
    return accuracy, precision, recall, f1, conf_matrix, roc_auc, fpr, tpr

results = []
roc_figures = []
for model in classifier:
    trained_model = education(model)
    accuracy, precision, recall, f1, conf_matrix, roc_auc, fpr, tpr = evaluate_model(trained_model, X_test, y_test)
    results.append((type(model).__name__, accuracy, precision, recall, f1, roc_auc))
    
    # ROC 
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {type(model).__name__}')
    plt.legend(loc="lower right")
    roc_filename = f'D:/Microsoft VS Code/Machine_Learning_Hw/roc_curve_{type(model).__name__}.png'
    plt.savefig(roc_filename)
    plt.close()
    roc_figures.append(roc_filename)

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score","ROC AUC"])
print(results_df.sort_values("Accuracy", ascending=False))



# Histogram of a single feature
plt.hist(data_copy['MonthlyCharges'], bins=30, edgecolor='k')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Monthly Charges')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/Histogram_plot.png')
plt.show()

# Scatter plot of two features
plt.scatter(data_copy['MonthlyCharges'], data_copy['TotalCharges'], alpha=0.5)
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.title('Monthly Charges vs Total Charges')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/Scatter_plot.png')
plt.show()

#It shows the distribution of the data and potential outliers.
# Box plot of a single feature
plt.boxplot(data_copy['MonthlyCharges'])
plt.xlabel('Monthly Charges')
plt.title('Box plot of Monthly Charges')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/Box_plot.png')
plt.show()

# Pair plot of the dataset
sns.pairplot(data_copy[['MonthlyCharges', 'TotalCharges', 'tenure']])
plt.title('Pair Plot of Monthly Charges, Total Charges, and Tenure')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/Pair_plot.png')
plt.show()

# Correlation matrix heatmap
corr_matrix = data_copy.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/Correlation_matrix.png')
plt.show() 

fig, ax = plt.subplots(figsize=(8, 2))  # Resmin boyutlarını ayarlayın
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/dataframe_image.png', bbox_inches='tight', dpi=200)

#churn Distribution w.r.t Gender
data.Churn[data.Churn == "No"].groupby(by = data.gender).count()
data.Churn[data.Churn == "Yes"].groupby(by = data.gender).count()
plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.savefig('D:/Microsoft VS Code/Machine_Learning_Hw/churndist_plot.png')
plt.show()


#Hyperparameter Tuning
# parametreler={
#     models[0]:{"C":[0.1,1,10,100],"penalty":["l1","l2"]},
#     models[1]:{"kernel":["linear", "rbf"],"C":[0.1,1],"gamma":[0.01,0.001]},
#     models[2]:{"alpha":[0.1,1.0]},
#     models[3]:{"C":[0.1,1],"penalty":["l1","l2"]},
#     models[4]:{"n_estimators":[1000,2000],"max_depth":[4,10], "min_samples_split":[2,5]},
#     models[5]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10], "subsample":[0.6,0.8]},
#     models[6]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10], "subsample":[0.6,0.8]},

# }


# for i,j in zip(models,sınıflar):
#     print(i)
#     grid=GridSearchCV(cozum(j), param_grid=parametreler[i], cv=10, n_jobs=-1)
#     grid.fit(X_train,y_train)
#     print(grid.best_params_)







                        
        
