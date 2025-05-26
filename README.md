# Brazilian E-Commerce Analysis & Seller Churn Prediction

**Contributors:** [Fritz Immanuel](https://github.com/fritz-immanuel), Gerard Louis Howan
**Project Type:** Data Science | Classification Modeling | Churn Analysis

> [!NOTE]
> Presentation Deck : <br>
> Dashboard : [Tableau](https://public.tableau.com/views/FinalProjectJCDS-2502Alpha_Group/SellerChurnOverview?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)<br>
> Streamlit Deploy : [Streamlit](https://alphagroup-jcds2502-finalproject-mdym3v3ideqvrfozwhtfxy.streamlit.app/)<br>

---

## 1. Business Context

[Olist](https://www.olist.com) is a leading Brazilian e-commerce enabler that helps small and medium-sized businesses sell through large marketplaces like Amazon, Magalu, and Mercado Libre. By offering end-to-end infrastructure—from logistics to customer management—Olist allows sellers to scale quickly without operational overhead.

However, as Olist scales, **seller churn** becomes a critical threat. Churned sellers reduce product availability, impact customer satisfaction, and weaken marketplace competitiveness. Worse, sellers often leave silently—stopping activity without formally exiting the platform.

## 2. Business Problem

Olist needs a way to **identify at-risk sellers before they churn**. Without a churn detection system:

- Seller attrition goes unnoticed
- Engagement teams can’t intervene in time
- Marketplace supply and revenue forecasting becomes unreliable

## 3. Project Goal

Build a **binary classification model** to predict whether a seller is:

- `1 = Churned` — No orders for **N+ days**
- `0 = Active` — Still fulfilling orders

> ⚠️ _Churn is defined empirically based on observed inactivity thresholds._

## 4. Key Questions

1. **What does the average seller lifecycle look like?**
2. **Do behavioral signals (like order volume, delays, or review scores) predict churn?**
3. **Are churn patterns linked to product category or payment method?**
4. **Can we reliably predict churn before it happens?**
5. **What features most influence churn, and how can we act on them?**

## 5. Analytical Approach

### Exploratory Analysis

- Visualized churn timing, cohort drop-offs, and seller order timelines
- Investigated review patterns, delivery delays, pricing, and category mix

### Churn Definition

- Churn defined as **45+ days of inactivity** after the last fulfilled order

### Feature Engineering

- Seller lifecycle duration, order velocity, delay stats, product diversity
- Review score distributions, category dominance, cohort tagging
- Early momentum (orders in first 34 days), payment method encoding

### Modeling

- Models tested: `LogisticRegression`, `KNN`, `DecisionTree`, `RandomForest`, `AdaBoost`, `GradientBoost`, `XGBoost`, `Voting`, `Stacking`
- Compared resampled (`RandomOverSampler`/`SMOTE`/`RandomUnderSampler`/`NearMiss`) and non-resampled performance

### Evaluation Strategy

- **F2 Score** used as the primary metric (recall-heavy)
- Grid and Randomized Search with Imbalanced Pipelines
- Confusion matrices used to interpret model tradeoffs

### Interpretation

- **Feature Importance** from Gradient Boost and SHAP
- Identified top churn drivers: delivery delays, review ratios, early order behavior

## 6. Results Summary

- **Best Model:** `Gradient Boost + SMOTE` (tuned)
- **F2 Score (Test):** `0.795` — high recall, acceptable precision
- **Key Drivers of Churn:**

  - `std_delivery_delay`, `avg_delivery_delay` — Late shipments drive churn
  - `payment_type_debit_card` — Indicates B2B-style sellers with higher churn
  - `early_orders_34d` — Low early activity correlates with drop-off
  - `cohort_month` and `top_category` — Seasonal or product-based churn clusters

## 7. Conclusion

The churn prediction model enables Olist to:

- Detect \~80% of likely churners before disengagement
- Prioritize retention resources and target risk groups
- Save an estimated **\$80k+ per quarter** in retained seller revenue
- Build a scalable, interpretable churn monitoring system

> This project highlights that **delivery performance, seller onboarding momentum, and cohort timing** are key to seller retention.

## 8. Future Recommendations

- Add seller-level support ticket data for complaint-based churn signals
- Analyze time-series drift and re-train monthly to preserve performance
- Test interventions (discounts, outreach) on predicted churners

## 9. Project Structure

```
📦 root
├── 📁 dataset/
|   ├──📁 cleaned/
|   ├──📁 raw/
|   └── ML_feature_set.csv
├── Data Analysis & Predictions.ipynb
├── 📁 models/
│   └── best_cv_result.pkl
├── 📁 streamlit/
└── README.md
```
