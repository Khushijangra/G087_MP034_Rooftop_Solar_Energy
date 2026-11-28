

\# Rooftop Solar Energy – Model Documentation



\*\*Project:\*\* G087 – MP034  

\*\*Role:\*\* Model Lead  

\*\*Contributor:\*\* Krishna Yadav  

\*\*Date:\*\* 29 November 2026



\## 1. Model Overview



The project predicts rooftop solar energy generation (`energy\_kwh`) using solar panel and environmental features.



The training pipeline uses:

\- Linear Regression

\- Ridge Regression (alpha = 0.1)

\- Ridge Regression (alpha = 1.0)



The preprocessing pipeline applies StandardScaler to numerical features and OneHotEncoder to categorical features.



\## 2. Model Evaluation Results



The following results were retrieved from the MLflow experiment `G087\_Solar\_Energy`.



| Model | Training MAE | Later MAE |

|---|---:|---:|

| Linear Regression | 1.498339 | 4.693370 |

| Ridge alpha 0.1 | 1.498334 | 4.693698 |

| Ridge alpha 1.0 | 1.498409 | 4.696650 |



MAE represents Mean Absolute Error. Lower values indicate smaller average absolute prediction errors.



\## 3. Champion Model Selection



The training pipeline selects the model with the lowest valid later MAE.



Linear Regression achieved the lowest later MAE of 4.693370 and was therefore selected as the champion model.



The selected model was:

\- Logged in MLflow.

\- Registered under `mp034-solar-model`.

\- Saved as `models/model.joblib` for inference.



The performance differences between the three models are small. Therefore, the results do not establish a substantial advantage over Ridge Regression.



\## 4. Model Limitations



1\. The model is trained using the available project dataset and may not generalize to every geographical or environmental condition.

2\. Unseen categorical values can cause prediction errors because the OneHotEncoder uses its default unknown-category handling.

3\. Prediction accuracy may be affected by changes in weather, dust accumulation, panel age and other environmental conditions.

4\. The available holdout prediction file contains record IDs and predictions but no actual target values. Therefore, holdout MAE, RMSE and R-squared cannot be calculated from that file alone.

5\. The current regression approach may not capture every nonlinear relationship affecting solar energy generation.



\## 5. Model Monitoring and Retraining Criteria



The following are recommended criteria for future monitoring and retraining:



\- Retrain when new verified solar energy observations become available.

\- Investigate a sustained increase in MAE on a labelled validation dataset.

\- Review the model when input feature distributions change significantly.

\- Re-evaluate performance when new locations, roof types or environmental conditions are introduced.

\- Compare the newly trained model with the current champion before replacing it.

\- Register a new champion only when its validation performance and reliability are acceptable.



These are proposed operational criteria, not automated triggers currently implemented in the project.



\## 6. Conclusion



The project compares Linear Regression and Ridge Regression using MLflow experiment tracking.



Linear Regression was selected as the champion because it achieved the lowest later MAE of 4.693370.



The champion model is saved as a Joblib artifact and is validated through automated inference tests.



This documentation records the model selection process, measured performance, limitations and recommended retraining criteria.



