# titanic_tutoring

# 🚢 Titanic - Machine Learning from Disaster

## 📌 კონკურსის მიმოხილვა
[Kaggle Titanic](https://www.kaggle.com/competitions/titanic) კონკურსის მიზანია იმის პროგნოზირება, გადარჩა თუ არა მგზავრი ტიტანიკის კატასტროფაში — ასაკის, სქესის, მგზავრის კლასის და სხვა მახასიათებლების საფუძველზე. ეს არის binary კლასიფიკაციის ამოცანა, რომელიც ფასდება **accuracy**-ს მიხედვით.

---

## 🧠 ჩემი მიდგომა
მონაცემების გაწმენდის შემდეგ ჩავატარე სხვადასხვა preprocessing ნაბიჯები (NA-ების შევსება, კოდირება, სვეტების მოცილება), დავწერე custom sklearn transformer კლასები და ავაგე ერთიანი sklearn `Pipeline`-ში. ყველა ექსპერიმენტი დავარეგისტრირე **MLflow**-ში **DagsHub**-ზე და გამოვიყენე **5-Fold Cross Validation** მოდელის სიზუსტის შესაფასებლად.

---

## 📁 რეპოზიტორიის სტრუქტურა

```
titanic_tutoring/
│
├── titanic-model-experiment.ipynb     ← EDA, preprocessing, ექსპერიმენტები
├── titanic-model-inference.ipynb      ← საუკეთესო მოდელის ჩამოტვირთვა, პროგნოზი, submission
├── README.md
```

---

## 📄 ფაილების აღწერა

| ფაილი | აღწერა |
|---|---|
| `titanic-model-experiment.ipynb`| მთავარი notebook EDA-სთვის, pipeline-ის აგებისა და მოდელების ექსპერიმენტებისთვის |
| `titanic-model-inference.ipynb` | MLflow-იდან pipeline-ისა და მოდელის ჩატვირთვა, Kaggle-ის ტესტ სეტის ტრანსფორმაცია და submission გენერაცია |

---

## 🧼 Cleaning

### ➤ მაღალი NA პროცენტის მქონე სვეტების მოცილება
სვეტები, სადაც missing value-ების პროცენტი ძალიან მაღალი იყო, მთლიანად დავდროპე:

- `Cabin` — მნიშვნელობების 77%-ზე მეტი იყო missing

### ➤ NA მნიშვნელობების შევსება
`ColumnTransformer` ამუშავებს შევსებას:
- **რიცხვითი სვეტები** (`Age`, `Fare`) → **მედიანით** შევსება
- **კატეგორიული სვეტები** (`Embarked`) → **მოდით** შევსება

### ➤ უსარგებლო სვეტების მოცილება
სვეტები, რომლებსაც პროგნოზული ღირებულება არ გააჩნდათ, დავდროპე feature selection-მდე:
- `Name`, `Ticket`, `PassengerId` — მხოლოდ იდენტიფიკატორებია, სიგნალი არ გააჩნიათ

---

## 🧬 Feature Engineering

### ➤ FamilySize
`SibSp` და `Parch`-ის ცალ-ცალკე გამოყენების ნაცვლად, გავაერთიანე ერთ უფრო ძლიერ feature-ში:

```python
FamilySize = SibSp + Parch + 1  # +1 თავად მგზავრის ჩასათვლელად
```

### ➤ კატეგორიული ცვლადების კოდირება
- **WOE (Weight of Evidence)** კოდირება გამოყენებულია `Pclass`-ზე — პირდაპირ ასახავს გადარჩენასთან კავშირს
- **OneHotEncoder** გამოყენებულია `Sex` და `Embarked`-ზე

---

## 🎯 Feature Selection

Pipeline-ში სამეტაპიანი feature selection გამოიყენება თანმიმდევრულად:

### ➤ RFE (Recursive Feature Elimination)
Logistic Regression გამოყენებულია base estimator-ად საუკეთესო 8 feature-ის შესარჩევად. LR კოეფიციენტები უფრო სტაბილური და ინტერპრეტირებადია Decision Tree-სთან შედარებით.

**RFE-ს მიერ შერჩეული feature-ები:**
```
ohe__Sex_female, ohe__Sex_male, ohe__Embarked_C, ohe__Embarked_S,
remainder__Pclass, remainder__SibSp, remainder__Parch, remainder__FamilySize
```

### ➤ კორელაციის ფილტრი
0.85-ზე მეტი წყვილური კორელაციის მქონე feature-ები მოიხსნა სიჭარბის შესამცირებლად.

**კორელაციის ფილტრის მიერ მოხსნილი:**
```
ohe__Sex_male, remainder__FamilySize
```

### ➤ IV (Information Value) ვალიდაცია
დარჩენილი feature-ები შემოწმდა Information Value-ს მიხედვით. 0.02-ზე დაბალი IV-ის მქონე feature-ები მოიხსნა.

**IV შედეგები:**
| Feature | IV | სიძლიერე |
|---|---|---|
| ohe__Sex_female | 1.3682 | ძლიერი |
| remainder__Pclass | 0.5224 | ძლიერი |
| remainder__SibSp | 0.1866 | საშუალო |
| remainder__Parch | 0.1429 | საშუალო |
| ohe__Embarked_C | 0.1329 | საშუალო |
| ohe__Embarked_S | 0.1327 | საშუალო |

ყველა 6 დარჩენილი feature გაიარა IV threshold-ს.

---

## ⚙️ სრული Pipeline

```
FamilySizeAdder          → FamilySize სვეტის დამატება
      ↓
ColumnTransformer        → Cabin-ის დროფი, Age/Fare-ის შევსება, Sex/Embarked-ის OHE, Name/Ticket/PassengerId-ის დროფი
      ↓
ColumnNameRestorer       → ColumnTransformer-ის შემდეგ დაკარგული სვეტის სახელების აღდგენა
      ↓
WOEEncoder               → Pclass-ის WOE კოდირება
      ↓
RFESelector              → საუკეთესო 8 feature-ის შერჩევა
      ↓
CorrelationFilter        → მაღალი კორელაციის მქონე feature-ების მოხსნა (threshold=0.85)
      ↓
IVValidator              → დაბალი IV-ის მქონე feature-ების მოხსნა (threshold=0.02)
```

---

## 🧪 ტრენინგი და ექსპერიმენტები

ყველა ექსპერიმენტი დარეგისტრირდა MLflow-ში DagsHub-ზე. თითოეული მოდელისთვის შემდეგი მეტრიკები დაილოგა:

- `train_accuracy`, `val_accuracy`, `test_accuracy`
- `train_f1`, `val_f1`, `test_f1`
- `train_roc_auc`, `val_roc_auc`, `test_roc_auc`
- `train_precision`, `val_precision`, `test_precision`
- `train_recall`, `val_recall`, `test_recall`

**შეფასების სტრატეგია:** 5-Fold Cross Validation `X_train`-ზე → საშუალო train/val შედეგები დაილოგა. შემდეგ სრული fit სრულ `X_train`-ზე → შეფასება held-out `X_test`-ზე.

---

### 🔹 Logistic Regression

შემოწმდა შემდეგი კომბინაციები:
- `C`: 0.01, 0.1, 1.0, 10.0
- `solver`: lbfgs, liblinear, saga
- `penalty`: l1, l2

**საუკეთესო Logistic Regression შედეგი:**

| მეტრიკა | შედეგი |
|---|---|
| val_accuracy | 0.7936 |
| test_accuracy | 0.8045 |
| val_roc_auc | 0.8210 |
| test_roc_auc | 0.8146 |
| val_f1 | 0.7158 |
| test_f1 | 0.7200 |

Train და validation შედეგები ძალიან ახლოს იყო ერთმანეთთან — overfitting არ დაფიქსირდა.

---

### 🔹 Decision Tree

შემოწმდა შემდეგი კომბინაციები:
- `max_depth`: 3, 5, 10, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4
- `criterion`: gini, entropy

მოსალოდნელი შედეგი დადასტურდა — დაბალი სიღრმის ხეებმა (`max_depth=3`, `max_depth=5`) უკეთ განაზოგადეს ვიდრე ღრმა ხეებმა. `max_depth=None`-მა აჩვენა გამოხატული overfitting — `train_accuracy` თითქმის 1.0-მდე, `val_accuracy` კი მნიშვნელოვნად დაეცა.

---

## ⭐️ საუკეთესო მოდელი

საუკეთესო მოდელი შეირჩა ორივე ექსპერიმენტის `val_accuracy`-ს შედარებით:

```python
mlflow.search_runs(
    experiment_names=["logistic_regression", "decision_tree"],
    order_by=["metrics.val_accuracy DESC"]
)
```

**შერჩევის კრიტერიუმი:** ყველაზე მაღალი `val_accuracy` სადაც `val_accuracy ≈ test_accuracy` — ანუ კარგი გენერალიზაცია და არა მხოლოდ შემთხვევითი split.

---

## 📊 MLflow ექსპერიმენტები DagsHub-ზე

ყველა run დარეგისტრირებულია:
👉 [dagshub.com/tvani2/titanic_tutoring](https://dagshub.com/tvani2/titanic_tutoring)

თითოეულ run-ში დაილოგა:
- ყველა ჰიპერპარამეტრი
- Train / Val / Test მეტრიკები (accuracy, f1, roc_auc, precision, recall)
- დატრენინგებული მოდელის არტეფაქტი (`.skops` / `.pkl`)
- Preprocessing pipeline-ის არტეფაქტი

---

## 🛠 გამოცდილება და გაკვეთილები

- **Preprocessing pipeline**-ის მოდელისგან გამოყოფა და ცალ-ცალკე MLflow-ში დარეგისტრირება inference-ს მარტივად გასაგებს ხდის
- **RFE → კორელაციის ფილტრი → IV** თანმიმდევრობა უფრო სუფთა feature selection-ს იძლევა ვიდრე ნებისმიერი ცალკეული მეთოდი
- ყოველთვის **train/test split უნდა მოხდეს EDA-მდე**
- **ტრენინგისა და inference-ის ცალკე notebook-ებში შენახვა** workflow-ს სუფთად ინახავს
- საუკეთესო მოდელის არჩევისას მხოლოდ **raw accuracy** არასაიმედო სიგნალია 
