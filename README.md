# Fraud Detection

| Column Name        | Meaning (Layman’s Terms)                                                         |
| ------------------ | -------------------------------------------------------------------------------- |
| accountNumber      | The bank account or card number used for the transaction.                        |
| merchantId         | A unique ID assigned to the merchant (shop, store, or business).                 |
| merchantZip        | The ZIP/postal code of the merchant’s location.                                  |
| merchantCountry    | The country where the merchant is located.                                       |
| posEntryMode       | How the transaction was made (e.g., Chip, Swipe, Online, Contactless).           |
| mcc                | Merchant Category Code (e.g., Grocery=5411, Gas Station=5541).                   |
| transactionAmount  | The amount of money spent in the transaction.                                    |
| transactionTime    | The date and time when the transaction happened.                                 |
| eventId            | A unique ID for each transaction (helps track specific purchases).               |
| fraudLabel         | 1 = Fraudulent transaction, 0 = Legitimate transaction.                          |


## Classifiers

- **KNeighborsClassifier**  
  Uses the k-nearest neighbors approach to classify transactions.
  
- **LogisticRegression**  
  A statistical model that estimates the probability of fraud.

- **RandomForestClassifier**  
  An ensemble of decision trees that often performs well in fraud detection.

- **AdaBoostClassifier**  
  Boosting method that combines weak learners to improve performance.

- **GradientBoostingClassifier**  
  Iteratively adds new models to correct errors made by existing ones.

- **XGBClassifier**  
  An optimized boosting framework known for speed and performance.

- **LGBMClassifier**  
  A gradient boosting model that uses tree-based learning with improved efficiency.

- **CatBoostClassifier**  
  Gradient boosting on decision trees with native handling of categorical features.

- **SVC**  
  Finds a decision boundary that separates classes with the largest margin.