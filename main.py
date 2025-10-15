import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Enhanced airline configuration with market share weights
AIRLINES = {
    "IndiGo": {"multiplier": 0.95, "weight": 40, "on_time": 0.82},
    "Air_India": {"multiplier": 1.05, "weight": 15, "on_time": 0.75},
    "SpiceJet": {"multiplier": 0.90, "weight": 15, "on_time": 0.78},
    "Vistara": {"multiplier": 1.20, "weight": 12, "on_time": 0.85},
    "AirAsia_India": {"multiplier": 0.88, "weight": 10, "on_time": 0.80},
    "Akasa_Air": {"multiplier": 0.92, "weight": 8, "on_time": 0.83},
}

# Major Indian cities with airport codes and base demand
CITIES = {
    "Delhi": {"code": "DEL", "demand": 1.3, "tier": 1},
    "Mumbai": {"code": "BOM", "demand": 1.3, "tier": 1},
    "Bangalore": {"code": "BLR", "demand": 1.25, "tier": 1},
    "Hyderabad": {"code": "HYD", "demand": 1.15, "tier": 1},
    "Chennai": {"code": "MAA", "demand": 1.15, "tier": 1},
    "Kolkata": {"code": "CCU", "demand": 1.1, "tier": 1},
    "Pune": {"code": "PNQ", "demand": 1.05, "tier": 2},
    "Ahmedabad": {"code": "AMD", "demand": 1.05, "tier": 2},
    "Goa": {"code": "GOI", "demand": 1.0, "tier": 2},
    "Jaipur": {"code": "JAI", "demand": 0.95, "tier": 2},
    "Kochi": {"code": "COK", "demand": 0.95, "tier": 2},
    "Chandigarh": {"code": "IXC", "demand": 0.9, "tier": 3},
    "Lucknow": {"code": "LKO", "demand": 0.9, "tier": 3},
    "Srinagar": {"code": "SXR", "demand": 0.85, "tier": 3},
    "Varanasi": {"code": "VNS", "demand": 0.85, "tier": 3},
    "Indore": {"code": "IDR", "demand": 0.88, "tier": 3},
    "Nagpur": {"code": "NAG", "demand": 0.85, "tier": 3},
    "Coimbatore": {"code": "CJB", "demand": 0.82, "tier": 3},
    "Patna": {"code": "PAT", "demand": 0.80, "tier": 3},
    "Bhubaneswar": {"code": "BBI", "demand": 0.80, "tier": 3},
}


# Route configuration with distance (km) and base pricing multiplier
def generate_routes():
    """Generate realistic routes between major Indian cities"""
    routes = []
    cities = list(CITIES.keys())

    # Tier 1 to Tier 1 routes (high frequency)
    tier1 = [c for c in cities if CITIES[c]["tier"] == 1]
    for i, src in enumerate(tier1):
        for dst in tier1[i + 1 :]:
            dist = random.randint(800, 2200)
            routes.append(
                {
                    "source": src,
                    "destination": dst,
                    "distance": dist,
                    "base_multiplier": 1.0,
                    "frequency": "high",
                }
            )

    # Tier 1 to Tier 2/3 routes (medium frequency)
    tier23 = [c for c in cities if CITIES[c]["tier"] in [2, 3]]
    for src in tier1:
        for dst in tier23:
            dist = random.randint(500, 2000)
            routes.append(
                {
                    "source": src,
                    "destination": dst,
                    "distance": dist,
                    "base_multiplier": 0.95,
                    "frequency": "medium",
                }
            )

    # Popular tourist routes
    tourist_routes = [
        ("Mumbai", "Goa", 440, 1.1, "high"),
        ("Delhi", "Srinagar", 660, 1.15, "medium"),
        ("Bangalore", "Goa", 520, 1.1, "medium"),
        ("Delhi", "Varanasi", 680, 1.0, "medium"),
        ("Bangalore", "Kochi", 490, 1.05, "medium"),
        ("Mumbai", "Jaipur", 920, 1.05, "medium"),
        ("Chennai", "Coimbatore", 340, 0.95, "high"),
        ("Delhi", "Lucknow", 500, 0.95, "high"),
        ("Mumbai", "Indore", 590, 1.0, "medium"),
    ]
    for src, dst, dist, mult, freq in tourist_routes:
        routes.append(
            {
                "source": src,
                "destination": dst,
                "distance": dist,
                "base_multiplier": mult,
                "frequency": freq,
            }
        )

    return routes


ROUTES = generate_routes()

# Time slots with realistic distribution
TIME_SLOTS = {
    "Early_Morning": {
        "start": "04:00",
        "end": "07:59",
        "weight": 15,
        "multiplier": 0.90,
    },
    "Morning": {"start": "08:00", "end": "11:59", "weight": 25, "multiplier": 1.05},
    "Afternoon": {"start": "12:00", "end": "15:59", "weight": 20, "multiplier": 1.0},
    "Evening": {"start": "16:00", "end": "19:59", "weight": 30, "multiplier": 1.15},
    "Night": {"start": "20:00", "end": "23:59", "weight": 10, "multiplier": 0.95},
}

# Indian festivals and holidays for 2025
SPECIAL_DATES = {
    "2025-01-01": {"name": "New Year", "boost": 1.4},
    "2025-01-14": {"name": "Makar Sankranti", "boost": 1.25},
    "2025-01-26": {"name": "Republic Day", "boost": 1.3},
    "2025-03-14": {"name": "Holi", "boost": 1.35},
    "2025-03-30": {"name": "Eid ul-Fitr", "boost": 1.3},
    "2025-04-10": {"name": "Ram Navami", "boost": 1.2},
    "2025-04-14": {"name": "Ambedkar Jayanti", "boost": 1.15},
    "2025-04-18": {"name": "Good Friday", "boost": 1.2},
    "2025-05-01": {"name": "May Day", "boost": 1.1},
    "2025-06-06": {"name": "Eid ul-Adha", "boost": 1.3},
    "2025-08-15": {"name": "Independence Day", "boost": 1.3},
    "2025-08-27": {"name": "Janmashtami", "boost": 1.2},
    "2025-10-02": {"name": "Gandhi Jayanti", "boost": 1.25},
    "2025-10-12": {"name": "Dussehra", "boost": 1.4},
    "2025-10-20": {"name": "Diwali", "boost": 1.5},
    "2025-10-21": {"name": "Diwali", "boost": 1.5},
    "2025-12-25": {"name": "Christmas", "boost": 1.35},
}


def get_seasonal_factor(date):
    """Advanced seasonal pricing based on Indian travel patterns"""
    month = date.month
    day = date.day

    # Peak seasons
    if month in [12, 1]:  # Winter holidays
        return 1.25
    elif month in [4, 5, 6]:  # Summer vacation
        return 1.20
    elif month in [10, 11]:  # Festive season (Diwali, etc.)
        return 1.30

    # Shoulder seasons
    elif month in [2, 3, 9]:
        return 1.05

    # Monsoon lull
    elif month in [7, 8]:
        return 0.85

    return 1.0


def get_day_of_week_factor(date):
    """Weekend and Friday premium"""
    weekday = date.weekday()
    if weekday == 4:  # Friday
        return 1.20
    elif weekday in [5, 6]:  # Saturday, Sunday
        return 1.25
    elif weekday == 0:  # Monday
        return 1.10
    return 1.0


def get_advance_booking_factor(days_before):
    """Realistic advance booking pricing curve"""
    if days_before <= 3:
        return 1.50  # Last minute
    elif days_before <= 7:
        return 1.30
    elif days_before <= 14:
        return 1.15
    elif days_before <= 30:
        return 1.0
    elif days_before <= 60:
        return 0.90
    else:
        return 0.85  # Early bird


def calculate_duration(distance, stops):
    """Calculate flight duration based on distance and stops"""
    base_speed = 750  # km/h average
    flight_time = distance / base_speed

    if stops == "zero":
        return round(flight_time + random.uniform(0, 0.2), 2)
    elif stops == "one":
        return round(flight_time + random.uniform(0.8, 1.5), 2)
    else:  # two_or_more
        return round(flight_time + random.uniform(1.5, 2.5), 2)


def generate_flight_number(airline, date):
    """Generate realistic flight numbers"""
    prefix = AIRLINES[airline].get("code", airline[:2].upper())
    number = random.randint(100, 9999)
    return f"{prefix}{number}"


def get_arrival_time(departure_slot):
    """Get realistic arrival time slot"""
    slots = list(TIME_SLOTS.keys())
    dep_idx = slots.index(departure_slot)
    # Most flights arrive 1-2 slots later
    arr_idx = min(dep_idx + random.choice([1, 2, 2]), len(slots) - 1)
    return slots[arr_idx]


def generate_dataset(year=2025, flights_per_day_range=(60, 100)):
    """Generate comprehensive flight dataset"""

    data = []
    start_date = datetime(year, 1, 1)
    num_days = 365

    # Weighted airline selection
    airline_names = list(AIRLINES.keys())
    airline_weights = [AIRLINES[a]["weight"] for a in airline_names]

    # Weighted time slot selection
    time_slots = list(TIME_SLOTS.keys())
    time_weights = [TIME_SLOTS[t]["weight"] for t in time_slots]

    print("🚀 Generating comprehensive Indian flight dataset...")

    for day_num in range(num_days):
        current_date = start_date + timedelta(days=day_num)
        date_str = current_date.strftime("%Y-%m-%d")

        # Variable flights per day based on day of week
        base_flights = random.randint(*flights_per_day_range)
        if current_date.weekday() in [4, 5, 6]:  # Weekend
            base_flights = int(base_flights * 1.4)
        if date_str in SPECIAL_DATES:  # Special dates get more flights
            base_flights = int(base_flights * 1.2)

        # Generate flights for this day
        for flight_num in range(base_flights):
            # Select route
            route = random.choice(ROUTES)

            # Select airline (weighted by market share)
            airline = random.choices(airline_names, weights=airline_weights)[0]
            airline_config = AIRLINES[airline]

            # Select departure time (weighted by popularity)
            dep_slot = random.choices(time_slots, weights=time_weights)[0]
            arr_slot = get_arrival_time(dep_slot)

            # Stops based on distance and route type
            if route["distance"] < 800:
                stops = "zero"
            elif route["distance"] < 1500:
                stops = random.choice(["zero", "zero", "one"])
            else:
                stops = random.choice(["zero", "one", "one"])

            # Duration
            duration = calculate_duration(route["distance"], stops)

            # Class
            cabin_class = random.choices(
                ["Economy", "Premium_Economy", "Business"], weights=[80, 12, 8]
            )[0]

            class_multiplier = {
                "Economy": 1.0,
                "Premium_Economy": 1.8,
                "Business": 3.5,
            }[cabin_class]

            # Advance booking (simulate booking window)
            days_before = random.choices(
                range(0, 91),
                weights=[5] * 3
                + [10] * 4
                + [15] * 7
                + [20] * 14
                + [15] * 21
                + [10] * 42,
            )[0]

            # Base price calculation (per km)
            base_price_per_km = 3.5
            base_price = route["distance"] * base_price_per_km

            # Apply all factors
            price = base_price
            price *= airline_config["multiplier"]
            price *= route["base_multiplier"]
            price *= TIME_SLOTS[dep_slot]["multiplier"]
            price *= class_multiplier
            price *= get_seasonal_factor(current_date)
            price *= get_day_of_week_factor(current_date)
            price *= get_advance_booking_factor(days_before)

            # Special date boost
            if date_str in SPECIAL_DATES:
                price *= SPECIAL_DATES[date_str]["boost"]

            # City demand factors
            price *= CITIES[route["source"]]["demand"]
            price *= CITIES[route["destination"]]["demand"]

            # Add realistic noise
            price *= random.uniform(0.92, 1.08)
            price = int(price)

            # Realistic price caps for Indian domestic flights
            if cabin_class == "Economy":
                price = max(min(price, 25000), 1500)
            elif cabin_class == "Premium_Economy":
                price = max(min(price, 40000), 3000)
            else:  # Business
                price = max(min(price, 60000), 8000)

            # Flight number
            flight_code = generate_flight_number(airline, current_date)

            data.append(
                {
                    "flight_id": len(data) + 1,
                    "date": date_str,
                    "airline": airline,
                    "flight_number": flight_code,
                    "source_city": route["source"],
                    "source_code": CITIES[route["source"]]["code"],
                    "destination_city": route["destination"],
                    "destination_code": CITIES[route["destination"]]["code"],
                    "departure_time": dep_slot,
                    "arrival_time": arr_slot,
                    "duration": duration,
                    "stops": stops,
                    "class": cabin_class,
                    "distance_km": route["distance"],
                    "price": price,
                    "days_before_departure": days_before,
                    "day_of_week": current_date.strftime("%A"),
                    "is_weekend": 1 if current_date.weekday() >= 5 else 0,
                    "is_holiday": 1 if date_str in SPECIAL_DATES else 0,
                    "month": current_date.month,
                    "quarter": (current_date.month - 1) // 3 + 1,
                    "season": get_season(current_date.month),
                    "booking_class": "Last_Minute"
                    if days_before <= 7
                    else "Advance"
                    if days_before >= 30
                    else "Normal",
                }
            )

        if (day_num + 1) % 30 == 0:
            print(f"✓ Processed {day_num + 1} days... ({len(data)} flights)")

    return pd.DataFrame(data)


def get_season(month):
    """Get Indian season"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post_Monsoon"


# Generate the dataset
df = generate_dataset(year=2025, flights_per_day_range=(80, 120))

# Create route column for analysis
df["route"] = df["source_city"] + " → " + df["destination_city"]

# Save to CSV
output_file = "indian_flights_2025_comprehensive.csv"
df.to_csv(output_file, index=False)

# Additional analytics - create summary CSV
summary_stats = {
    "Total Flights": [len(df)],
    "Avg Flights Per Day": [len(df) / 365],
    "Total Routes": [df["route"].nunique()],
    "Avg Price": [f"₹{df['price'].mean():.0f}"],
    "Date Range": [f"{df['date'].min()} to {df['date'].max()}"],
}
pd.DataFrame(summary_stats).to_csv("dataset_summary.csv", index=False)

# Generate summary statistics
print("\n" + "=" * 60)
print("📊 DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Total flights: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Average flights per day: {len(df) / 365:.1f}")
print(f"Unique routes: {df['route'].nunique()}")
print(f"Cities covered: {len(CITIES)}")
print(f"\n✈️ Airlines Distribution:")
for airline in df["airline"].value_counts().items():
    pct = (airline[1] / len(df)) * 100
    print(f"  - {airline[0]}: {airline[1]:,} flights ({pct:.1f}%)")
print(f"\n🎫 Class Distribution:")
for cls in df["class"].value_counts().items():
    pct = (cls[1] / len(df)) * 100
    print(f"  - {cls[0]}: {cls[1]:,} flights ({pct:.1f}%)")
print(f"\n🏙️ Top 10 Busiest Routes:")
for route, count in df["route"].value_counts().head(10).items():
    print(f"  - {route}: {count:,} flights")
print(f"\n💰 Price Statistics:")
print(f"  - Mean: ₹{df['price'].mean():,.0f}")
print(f"  - Median: ₹{df['price'].median():,.0f}")
print(f"  - Min: ₹{df['price'].min():,}")
print(f"  - Max: ₹{df['price'].max():,}")
print(f"  - Economy Avg: ₹{df[df['class'] == 'Economy']['price'].mean():,.0f}")
print(f"  - Business Avg: ₹{df[df['class'] == 'Business']['price'].mean():,.0f}")
print(f"\n📅 Peak Travel Days:")
daily_counts = df.groupby("date").size().sort_values(ascending=False).head(5)
for date, count in daily_counts.items():
    day_info = df[df["date"] == date].iloc[0]
    holiday = SPECIAL_DATES.get(date, {}).get(
        "name", "Weekend" if day_info["is_weekend"] else ""
    )
    print(
        f"  - {date} ({day_info['day_of_week']}): {count} flights {f'[{holiday}]' if holiday else ''}"
    )
print(f"\n💾 Files saved:")
print(f"  - {output_file}")
print(f"  - dataset_summary.csv")
print("=" * 60)
