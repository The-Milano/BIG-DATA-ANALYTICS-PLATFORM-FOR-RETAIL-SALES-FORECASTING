"""
NEXUS RETAIL AI - Customer Intelligence & RFM Engine
Computes Recency, Frequency, Monetary (RFM) metrics, feature standardization,
K-Means clustering (k=3 to 6), 3D coordinate mapping, churn probability, and marketing playbooks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
try:
    from .data_engine import data_engine
except (ImportError, ValueError):
    from data_engine import data_engine


# Canonical segment profiles for k=4
ARCHETYPE_LABELS_4 = {
    0: ("Champions", "High spenders with recent and frequent repeat purchases.", "Send VIP Loyalty Code & early access to premium product drops."),
    1: ("Loyal Customers", "Consistent regular shoppers with strong monetary value.", "Upsell complimentary categories and offer tier-based rewards."),
    2: ("At-Risk Customers", "High-value shoppers who haven't ordered recently.", "Deploy automated win-back campaign with 15% limited-time discount."),
    3: ("Lost / Hibernating", "Infrequent purchasers with high recency lapse.", "Execute re-engagement survey and low-frequency digest emails."),
}

# General naming template for k=3 to k=6
ARCHETYPE_PRESETS = {
    3: [
        ("High-Value Champions", "Highest monetary value and high frequency.", "Provide concierge support & VIP rewards."),
        ("Steady Regulars", "Moderate frequency and consistent order volume.", "Cross-sell trending items & loyalty points."),
        ("Dormant / Lapsed", "High recency lapse with low engagement.", "Automate reactivation promo discount."),
    ],
    4: [
        ("Champions", "Highest frequency, lowest recency, top monetary spend.", "Send VIP Loyalty Code & priority drops."),
        ("Loyal Customers", "Consistent regular shoppers with strong lifetime value.", "Upsell complimentary categories."),
        ("At-Risk Customers", "High past spend but no orders in recent months.", "Trigger personalized win-back offer."),
        ("Lost / Hibernating", "Lowest activity and long lapse time.", "Re-engagement email digest & discount."),
    ],
    5: [
        ("Platinum Champions", "Elite top 5% revenue drivers and advocates.", "Bespoke executive concierge & private previews."),
        ("Loyal Regulars", "Reliable multi-category frequent buyers.", "Encourage referral bonuses and subscription perks."),
        ("Potential Loyalists", "Recent shoppers with high initial spend.", "Nurture with product education and onboarding."),
        ("At-Risk High Spenders", "Past whales who have stopped purchasing.", "Direct account manager outreach & win-back coupon."),
        ("Hibernating Low Value", "Old single-purchase shoppers with low basket.", "Low-cost programmatic re-targeting."),
    ],
    6: [
        ("Elite Champions", "Top spenders, high purchase velocity.", "Exclusive VIP previews & bespoke service."),
        ("Core Loyalists", "Predictable weekly/monthly high spenders.", "Tiered loyalty gamification & reward points."),
        ("New High Potentials", "Recent large basket first/second timers.", "Welcome bundle discounts & cross-sell."),
        ("Need Attention", "Above-average spenders nearing lapse threshold.", "Limited-time flash sale incentives."),
        ("At-Risk Lapsed", "Historically profitable accounts drifting away.", "Aggressive win-back campaigns."),
        ("Lost Inactive", "Zero activity in over 180 days.", "Automated quarterly reactivation pulse."),
    ],
}


class CustomerEngine:
    """Customer analytics engine for RFM, K-Means clustering, and churn estimation."""

    def __init__(self):
        self.data_engine = data_engine
        self._cached_rfm: Optional[pd.DataFrame] = None
        self._last_k: Optional[int] = None
        self._cached_result: Optional[Dict[str, Any]] = None

    def compute_rfm_table(self) -> pd.DataFrame:
        """
        Calculates raw Recency, Frequency, and Monetary metrics for each unique customer.
        - Recency: Days since customer's last purchase relative to reference dataset max date.
        - Frequency: Count of unique transactions / invoices.
        - Monetary: Total sum of purchases ($).
        """
        if self._cached_rfm is not None:
            return self._cached_rfm

        df = self.data_engine.load_data()
        ref_date = df["Timestamp"].max()

        rfm = (
            df.groupby("CustomerID")
            .agg(
                Recency=("Timestamp", lambda dates: (ref_date - dates.max()).days),
                Frequency=("InvoiceID", "nunique"),
                Monetary=("Total_Amount", "sum"),
                FirstPurchase=("Timestamp", "min"),
                LastPurchase=("Timestamp", "max"),
                AvgBasket=("Total_Amount", "mean"),
            )
            .reset_index()
        )

        rfm["Recency"] = rfm["Recency"].astype(int)
        rfm["Frequency"] = rfm["Frequency"].astype(int)
        rfm["Monetary"] = rfm["Monetary"].round(2)
        rfm["AvgBasket"] = rfm["AvgBasket"].round(2)

        # Churn Probability Model:
        # Heuristic sigmoid based on recency vs expected purchase interval
        # If recency > 3 * (tenure / frequency), churn risk is high
        tenure_days = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
        avg_interval = (tenure_days / np.maximum(rfm["Frequency"] - 1, 1)).clip(lower=7, upper=120)
        lapse_ratio = rfm["Recency"] / avg_interval
        rfm["Churn_Probability"] = (1.0 / (1.0 + np.exp(-1.5 * (lapse_ratio - 1.8)))).clip(0.05, 0.98).round(3)

        self._cached_rfm = rfm
        return rfm

    def perform_segmentation(self, k: int = 4, sample_limit: int = 2500) -> Dict[str, Any]:
        """
        Performs K-Means clustering on standardized RFM features:
        - Log transformation to handle skewed monetary and frequency values
        - StandardScaler normalization
        - Returns 3D coordinates for Plotly, segment summaries, and marketing action table
        """
        k = max(3, min(k, 6))

        if self._cached_result is not None and self._last_k == k:
            return self._cached_result

        rfm = self.compute_rfm_table().copy()

        # Log transform + Standardize
        rfm_log = pd.DataFrame()
        rfm_log["Recency"] = np.log1p(rfm["Recency"])
        rfm_log["Frequency"] = np.log1p(rfm["Frequency"])
        rfm_log["Monetary"] = np.log1p(rfm["Monetary"])

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # Sort clusters by monetary value to assign consistent semantic archetype labels
        cluster_monetary = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
        rank_to_cluster = list(cluster_monetary.index)
        preset_archetypes = ARCHETYPE_PRESETS.get(k, ARCHETYPE_PRESETS[4])

        cluster_meta = {}
        for rank_idx, orig_c in enumerate(rank_to_cluster):
            name, desc, action = preset_archetypes[rank_idx]
            cluster_meta[orig_c] = {
                "name": name,
                "description": desc,
                "action": action,
                "rank": rank_idx,
            }

        rfm["Segment_Name"] = rfm["Cluster"].map(lambda c: cluster_meta[c]["name"])
        rfm["Marketing_Action"] = rfm["Cluster"].map(lambda c: cluster_meta[c]["action"])

        # Segment Profile Summary Table
        summary_rows = []
        total_revenue = rfm["Monetary"].sum()
        total_customers = len(rfm)

        for orig_c in rank_to_cluster:
            c_df = rfm[rfm["Cluster"] == orig_c]
            meta = cluster_meta[orig_c]
            c_count = len(c_df)
            c_rev = float(c_df["Monetary"].sum())
            c_share = round((c_rev / total_revenue) * 100, 1)
            c_pct_cust = round((c_count / total_customers) * 100, 1)

            summary_rows.append({
                "cluster_id": int(orig_c),
                "segment_name": meta["name"],
                "customer_count": c_count,
                "customer_pct": c_pct_cust,
                "revenue_total": round(c_rev, 2),
                "revenue_share_pct": c_share,
                "mean_recency_days": round(float(c_df["Recency"].mean()), 1),
                "mean_frequency": round(float(c_df["Frequency"].mean()), 1),
                "mean_monetary": round(float(c_df["Monetary"].mean()), 2),
                "mean_aov": round(float(c_df["AvgBasket"].mean()), 2),
                "avg_churn_risk": round(float(c_df["Churn_Probability"].mean()) * 100, 1),
                "marketing_action": meta["action"],
                "description": meta["description"],
            })

        # Subsample for smooth 3D Plotly rendering
        if len(rfm) > sample_limit:
            rfm_sample = rfm.sample(n=sample_limit, random_state=42)
        else:
            rfm_sample = rfm

        scatter_points = []
        for _, row in rfm_sample.iterrows():
            scatter_points.append({
                "customer_id": row["CustomerID"],
                "recency": int(row["Recency"]),
                "frequency": int(row["Frequency"]),
                "monetary": float(row["Monetary"]),
                "cluster": int(row["Cluster"]),
                "segment_name": row["Segment_Name"],
                "churn_risk": round(float(row["Churn_Probability"]) * 100, 1),
                "avg_basket": float(row["AvgBasket"]),
            })

        result = {
            "k": k,
            "segments": summary_rows,
            "scatter_3d": scatter_points,
            "total_customers": total_customers,
            "total_revenue": round(total_revenue, 2),
        }

        self._last_k = k
        self._cached_result = result
        return result


# Global singleton instance
customer_engine = CustomerEngine()
