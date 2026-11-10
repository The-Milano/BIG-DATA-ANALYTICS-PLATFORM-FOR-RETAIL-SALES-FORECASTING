"""
NEXUS RETAIL AI - Data Engineering Engine
Synthesizes and manages 100,000+ realistic retail transaction rows with
seasonal variability, store-level trends, customer purchasing cycles, and holiday spikes.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "retail_transactions.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "retail_transactions.parquet")

# Product Catalog Hierarchy
CATALOG = {
    "Electronics": [
        ("Apex Pro 16 Laptop", 1299.00),
        ("Aura Wireless Noise-Cancelling Headphones", 249.99),
        ("UltraVision 34-Inch Curved OLED Monitor", 699.99),
        ("PulseFit Smartwatch Ultra", 199.50),
        ("Quantum 65W GaN Fast Charger", 49.99),
        ("Nova 4K Streaming Hub", 89.00),
    ],
    "Fashion & Apparel": [
        ("Merino Wool Cashmere Sweater", 145.00),
        ("Urban Tailored Denim Jeans", 89.50),
        ("Waterproof Trench Overcoat", 220.00),
        ("CloudStep Everyday Luxury Sneakers", 130.00),
        ("Classic Oxford Cotton Shirt", 65.00),
        ("Thermal Insulated Puffer Jacket", 175.00),
    ],
    "Home & Living": [
        ("ErgoComfort Executive Mesh Chair", 389.00),
        ("Artisan Barista Espresso Machine", 499.00),
        ("PureBreeze HEPA Smart Air Purifier", 159.00),
        ("Organic Egyptian Cotton Bedding Set", 129.50),
        ("Minimalist Nordic Floor Lamp", 85.00),
        ("HydroClean Robotic Vacuum Cleaner", 299.00),
    ],
    "Beauty & Health": [
        ("Bio-Luminescent Retinol Night Serum", 68.00),
        ("Ionic Ceramic Fast Hair Dryer", 125.00),
        ("Marine Collagen Peptides 500g", 45.00),
        ("Botanical Radiance Facial Cream", 52.00),
        ("Sonic Rechargeable Electric Toothbrush", 79.99),
    ],
    "Sports & Outdoors": [
        ("TerraGrip All-Weather Hiking Boots", 165.00),
        ("Zenith Non-Slip Pro Yoga Mat", 42.00),
        ("CarbonFiber Aero Road Helmet", 119.00),
        ("Alpine Expedition 3-Person Tent", 279.00),
        ("HydraFlow Insulated 1L Flask", 34.00),
    ],
    "Gourmet Grocery": [
        ("Single-Origin Ethiopian Yirgacheffe Beans", 22.50),
        ("Cold-Pressed Extra Virgin Tuscan Olive Oil", 32.00),
        ("Raw Manuka Honey MGO 400+", 48.00),
        ("Artisanal Dark Chocolate Collection", 18.50),
        ("Aged Balsamic Vinegar of Modena 250ml", 28.00),
    ],
}

STORES = [
    "New York - 5th Ave Flagship",
    "San Francisco - Union Square",
    "Chicago - Magnificent Mile",
    "Austin - Domain Tech Corridor",
    "London - Oxford Street",
    "Tokyo - Shibuya Crossing",
]

STORE_WEIGHTS = [0.24, 0.20, 0.18, 0.15, 0.13, 0.10]

PAYMENT_METHODS = [
    "Credit Card",
    "Apple Pay / Mobile Wallet",
    "Buy Now Pay Later (BNPL)",
    "Cash",
    "Cryptocurrency",
]
PAYMENT_WEIGHTS = [0.44, 0.32, 0.12, 0.08, 0.04]


class DataEngine:
    """Enterprise-grade data generator and manager for Nexus Retail AI."""

    def __init__(self, data_path: str = CSV_PATH):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def generate_synthetic_data(
        self,
        n_rows: int = 120_000,
        start_date: str = "2024-11-01",
        end_date: str = "2026-10-31",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates 100,000+ realistic transaction records with:
        - Multi-year timeline (2024-2026)
        - Q4 Holiday surge (Black Friday, Cyber Week, Christmas)
        - Day-of-week seasonality (Weekend surges)
        - Diurnal intraday curves (peak lunch & evening shopping)
        - Power-law customer spending frequency (for realistic RFM)
        """
        np.random.seed(seed)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days + 1

        # Flatten catalog products
        all_products = []
        all_categories = []
        all_prices = []
        for cat, items in CATALOG.items():
            for name, price in items:
                all_products.append(name)
                all_categories.append(cat)
                all_prices.append(price)

        num_catalog_items = len(all_products)
        cat_map = {name: cat for name, cat in zip(all_products, all_categories)}
        price_map = {name: price for name, price in zip(all_products, all_prices)}

        # 1. Generate Timestamps with Seasonality
        # Create daily transaction volume distribution
        daily_dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
        daily_weights = []

        for d in daily_dates:
            w = 1.0
            # Day of week seasonality: Friday, Saturday, Sunday boost
            dow = getattr(d, 'day_of_week', getattr(d, 'dayofweek', 0))
            if dow == 5:  # Saturday
                w *= 1.42
            elif dow == 6:  # Sunday
                w *= 1.30
            elif dow == 4:  # Friday
                w *= 1.18
            else:
                w *= 0.88

            # Annual Trend & Inflation growth (+8% YoY)
            days_from_start = (d - start_dt).days
            w *= (1.0 + 0.08 * (days_from_start / 365.25))

            # Seasonal Spikes: Q4 boost & Holidays
            month, day = d.month, d.day
            if month == 11 and 20 <= day <= 29:  # Black Friday / Cyber Week
                w *= 2.9
            elif month == 12 and 10 <= day <= 24:  # Christmas shopping rush
                w *= 2.4
            elif month == 12 and 26 <= day <= 31:  # Year-end clearance
                w *= 1.8
            elif month == 1:  # Post-holiday slump
                w *= 0.78
            elif month == 2 and 10 <= day <= 14:  # Valentine's Day
                w *= 1.35
            elif month == 7 and 10 <= day <= 20:  # Mid-Year Summer Mega Sale
                w *= 1.45
            elif month in [11, 12]:  # Overall Q4 lift
                w *= 1.35

            daily_weights.append(w)

        daily_weights = np.array(daily_weights)
        daily_probs = daily_weights / daily_weights.sum()

        chosen_day_indices = np.random.choice(len(daily_dates), size=n_rows, p=daily_probs)
        chosen_dates = daily_dates[chosen_day_indices]

        # Intra-day hour distribution (store open 8am to 10pm, peak 12-2pm and 6-8pm)
        hour_weights = [
            0.01, 0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04,  # 0-7
            0.08, 0.12, 0.16, 0.22, 0.32, 0.30, 0.25, 0.24,  # 8-15
            0.28, 0.34, 0.38, 0.31, 0.22, 0.14, 0.06, 0.02   # 16-23
        ]
        hour_probs = np.array(hour_weights) / sum(hour_weights)
        chosen_hours = np.random.choice(24, size=n_rows, p=hour_probs)
        chosen_minutes = np.random.randint(0, 60, size=n_rows)
        chosen_seconds = np.random.randint(0, 60, size=n_rows)

        # Vectorized timestamp construction
        timestamps = chosen_dates + pd.to_timedelta(chosen_hours, unit="h") + \
                     pd.to_timedelta(chosen_minutes, unit="m") + \
                     pd.to_timedelta(chosen_seconds, unit="s")

        # Sort chronologically
        sort_order = np.argsort(timestamps)
        timestamps = timestamps[sort_order]

        # 2. Customer IDs (Power Law distribution: ~12,500 unique customers)
        n_customers = 12_500
        # Pareto distribution creates Champions (many purchases) vs At-risk / One-timers
        pareto_samples = np.random.pareto(a=1.4, size=n_rows)
        customer_indices = (pareto_samples % n_customers).astype(int)
        customer_ids = [f"CUST-{idx+10001:05d}" for idx in customer_indices]

        # 3. Store Locations
        stores = np.random.choice(STORES, size=n_rows, p=STORE_WEIGHTS)

        # 4. Product Catalog Selection
        # Category preference skew slightly towards Electronics & Fashion
        product_probs = []
        for cat, items in CATALOG.items():
            base_p = 1.0
            if cat in ["Electronics", "Fashion & Apparel"]:
                base_p = 1.4
            for _ in items:
                product_probs.append(base_p)
        product_probs = np.array(product_probs) / sum(product_probs)

        prod_indices = np.random.choice(num_catalog_items, size=n_rows, p=product_probs)
        product_names = [all_products[i] for i in prod_indices]
        categories = [all_categories[i] for i in prod_indices]
        base_unit_prices = np.array([all_prices[i] for i in prod_indices])

        # Realistic Unit Price fluctuation (+/- 2% dynamic promotional pricing)
        price_noise = np.random.uniform(0.98, 1.02, size=n_rows)
        unit_prices = np.round(base_unit_prices * price_noise, 2)

        # 5. Quantity (1 to 5, Poisson-like)
        quantities = np.random.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.64, 0.23, 0.08, 0.03, 0.02])

        # 6. Discount Rates (skewed towards 0%, with occasional promo discounts)
        # Check if timestamp is in holiday spike for higher discount probability
        is_holiday_season = (timestamps.month.isin([11, 12])) | (timestamps.month == 7)
        discount_candidates = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
        regular_p = [0.68, 0.14, 0.10, 0.05, 0.02, 0.01]
        holiday_p = [0.42, 0.20, 0.18, 0.12, 0.06, 0.02]

        discounts = np.zeros(n_rows, dtype=float)
        reg_mask = ~is_holiday_season
        hol_mask = is_holiday_season

        discounts[reg_mask] = np.random.choice(discount_candidates, size=reg_mask.sum(), p=regular_p)
        discounts[hol_mask] = np.random.choice(discount_candidates, size=hol_mask.sum(), p=holiday_p)

        # 7. Total Amount
        total_amounts = np.round(quantities * unit_prices * (1.0 - discounts), 2)

        # 8. Payment Method
        payment_methods = np.random.choice(PAYMENT_METHODS, size=n_rows, p=PAYMENT_WEIGHTS)

        # 9. Invoice ID
        invoice_ids = [f"INV-{idx+1000001:07d}" for idx in range(n_rows)]

        df = pd.DataFrame({
            "InvoiceID": invoice_ids,
            "Timestamp": timestamps,
            "CustomerID": customer_ids,
            "Store_Location": stores,
            "Product_Category": categories,
            "Product_Name": product_names,
            "Quantity": quantities,
            "UnitPrice": unit_prices,
            "Discount": discounts,
            "Total_Amount": total_amounts,
            "Payment_Method": payment_methods,
        })

        # Save to disk
        self._save_data(df)
        self.df = df
        return df

    def _save_data(self, df: pd.DataFrame):
        """Persists dataset to CSV and high-speed Parquet."""
        self._ensure_data_dir()
        df.to_csv(CSV_PATH, index=False)
        try:
            df.to_parquet(PARQUET_PATH, index=False)
        except Exception:
            pass  # Non-fatal if pyarrow parquet engine is unavailable

    def load_data(self, force_regenerate: bool = False, min_rows: int = 100_000) -> pd.DataFrame:
        """
        Loads cached dataset or synthesizes on demand if missing or too small.
        Sub-second loading using Parquet when available, falling back to CSV.
        """
        if self.df is not None and not force_regenerate:
            return self.df

        if not force_regenerate:
            # Try fast Parquet load first
            if os.path.exists(PARQUET_PATH):
                try:
                    df = pd.read_parquet(PARQUET_PATH)
                    if len(df) >= min_rows:
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                        self.df = df
                        return self.df
                except Exception:
                    pass

            # Fallback to CSV load
            if os.path.exists(CSV_PATH):
                try:
                    df = pd.read_csv(CSV_PATH)
                    if len(df) >= min_rows:
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                        self.df = df
                        return self.df
                except Exception:
                    pass

        # Generate fresh dataset if not present
        return self.generate_synthetic_data(n_rows=120_000)

    def get_filter_metadata(self) -> Dict[str, Any]:
        """Returns unique categories, store locations, and date ranges."""
        df = self.load_data()
        return {
            "stores": sorted(df["Store_Location"].unique().tolist()),
            "categories": sorted(df["Product_Category"].unique().tolist()),
            "min_date": str(df["Timestamp"].min().date()),
            "max_date": str(df["Timestamp"].max().date()),
            "total_records": len(df),
            "unique_customers": int(df["CustomerID"].nunique()),
            "unique_skus": int(df["Product_Name"].nunique()),
        }


# Global singleton instance
data_engine = DataEngine()


if __name__ == "__main__":
    t0 = time.time()
    print("Generating Nexus Retail synthetic dataset...")
    df_gen = data_engine.generate_synthetic_data(n_rows=120_000)
    t1 = time.time()
    print(f"Generated {len(df_gen):,} rows in {t1 - t0:.2f} seconds.")
    print("Columns:", df_gen.columns.tolist())
    print("Date range:", df_gen['Timestamp'].min(), "to", df_gen['Timestamp'].max())
    print("Total revenue: ${:,.2f}".format(df_gen['Total_Amount'].sum()))
