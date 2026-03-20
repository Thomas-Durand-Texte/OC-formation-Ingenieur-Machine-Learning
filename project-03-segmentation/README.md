# E-commerce Customer Segmentation

**RFM Analysis and Clustering for Targeted Marketing**

---

## 🎯 Objective

Segment e-commerce customers based on purchasing behavior to enable personalized marketing strategies. Using RFM (Recency, Frequency, Monetary) analysis combined with unsupervised learning, identify distinct customer groups with actionable characteristics for targeted campaigns.

## 📊 Dataset

**Source**: UK-based online retail transaction data

**Period**: 12 months of transaction history

**Size**: ~500,000 transactions from ~4,000 customers

**Key Features**:
- **Transaction data**: Invoice date, product, quantity, price
- **Customer data**: Customer ID, country
- **Temporal**: Order timestamps for recency calculation
- **Financial**: Transaction amounts for monetary value

## 🧠 Approach

### Part 1: Data Exploration (`01_exploration.ipynb`)

**Data Understanding**:
- Transaction patterns over time (seasonality, trends)
- Product category analysis
- Customer geographic distribution
- Missing data and anomaly detection (returns, cancellations)

**Preprocessing**:
- Handling negative quantities (returns)
- Outlier detection and treatment
- Data cleaning and validation
- Feature engineering for RFM metrics

### Part 2: RFM Segmentation (`02_rfm_segmentation.ipynb`)

**RFM Metric Calculation**:
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Number of transactions in period
- **Monetary (M)**: Total spending amount

![RFM Distributions](figures/Recence.svg)

**Clustering Approaches**:
- K-means clustering on standardized RFM features
- Hierarchical clustering for dendrogram analysis
- DBSCAN for density-based segments
- Optimal cluster selection using:
  - Elbow method (inertia)
  - Silhouette score
  - Davies-Bouldin index

**Segment Profiling**:
- Statistical characterization of each segment
- Polar plots for multi-dimensional visualization
- Business interpretation of segments

![Customer Segments](figures/polar_centers.svg)

## 📈 Results

### Customer Segments Identified

**Optimal Segmentation**: 4-5 distinct customer groups

#### Segment Profiles

1. **Champions** (High R, High F, High M)
   - Recent, frequent, high-value customers
   - ~15-20% of customer base
   - Action: VIP treatment, exclusive offers, loyalty rewards

2. **Loyal Customers** (Mid R, High F, Mid-High M)
   - Regular purchasers with consistent spending
   - ~20-25% of customer base
   - Action: Engagement campaigns, cross-sell opportunities

3. **Potential Loyalists** (High R, Low-Mid F, Mid M)
   - Recent customers with growth potential
   - ~25-30% of customer base
   - Action: Nurture campaigns, product recommendations

4. **At Risk** (Low R, High F, High M)
   - Previously valuable but becoming inactive
   - ~10-15% of customer base
   - Action: Re-engagement campaigns, win-back offers

5. **Hibernating** (Low R, Low F, Low M)
   - Inactive, low-value customers
   - ~20-30% of customer base
   - Action: Low-cost reactivation or remove from active campaigns

### Model Performance

**Clustering Quality**:
- Silhouette Score: 0.45-0.55 (moderate to good separation)
- Clear segment differentiation in RFM space
- Interpretable business segments

### Key Insights

**Customer Behavior Patterns**:
- Strong concentration of customers in low-frequency, low-monetary segments
- Recency highly predictive of future purchase probability
- Frequency and Monetary value strongly correlated
- Geographic patterns in segment distribution

**Business Value**:
- Clear prioritization for marketing resources
- Quantified value of different customer segments
- Actionable retention and acquisition strategies

## 🔑 Key Learnings

**Machine Learning Skills**:
- RFM analysis for customer segmentation
- Multiple clustering algorithms (K-means, hierarchical, DBSCAN)
- Cluster validation metrics and selection
- Feature scaling and standardization
- Dimensionality reduction for visualization

**Business Analytics**:
- Customer lifetime value estimation
- Churn risk identification
- Segment-specific marketing strategy development
- Pareto principle in e-commerce (80/20 rule)

**Practical Applications**:
- **Champions**: Exclusive access, premium support, referral programs
- **At Risk**: Personalized win-back emails, special discounts
- **Potential Loyalists**: Product education, subscription offers
- **Hibernating**: Minimize spend, periodic low-cost reactivation

## 📁 Project Structure

```
project-04-segmentation/
├── README.md                    # This file
├── 01_exploration.ipynb         # Data exploration and preprocessing
├── 02_rfm_segmentation.ipynb    # RFM analysis and clustering
└── figures/                     # Visualizations
    ├── Recence.svg              # Recency distribution
    ├── Frequence.svg            # Frequency distribution
    ├── Montant.svg              # Monetary value distribution
    ├── polar_centers.svg        # Segment profiles (radar chart)
    ├── pie.svg                  # Segment size distribution
    └── dendrograms/             # Hierarchical clustering trees
```

## 🛠️ Technologies

- **Python 3.x**
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn
  - K-means clustering
  - Hierarchical clustering
  - DBSCAN
  - StandardScaler for feature normalization
- **Evaluation**: Silhouette score, Davies-Bouldin index, elbow method
- **Visualization**: Matplotlib, Seaborn, polar plots

---

**Date**: April 2023
**Training Program**: OpenClassrooms × CentraleSupélec — Machine Learning Engineer
