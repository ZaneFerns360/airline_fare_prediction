from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime, date
from enum import Enum
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import uvicorn

# ============================================================================
# INITIALIZE SPARK AND LOAD MODEL
# ============================================================================
print("🚀 Initializing Flight Price Prediction API...")

spark = (
    SparkSession.builder.master("local[*]")
    .appName("FlightPricePredictionAPI")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load model
MODEL_PATH = "flight_price_prediction_model"
print(f"📦 Loading model from {MODEL_PATH}...")
model = PipelineModel.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load metadata
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

VALID_AIRLINES = metadata["valid_airlines"]
VALID_CITIES = metadata["valid_cities"]

print(f"✓ Loaded {len(VALID_AIRLINES)} airlines and {len(VALID_CITIES)} cities")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================
app = FastAPI(
    title="✈️ Flight Price Prediction API",
    description="Predict Indian domestic flight prices using ML model trained on 40K+ flights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENUMS FOR VALIDATION
# ============================================================================
class DepartureTime(str, Enum):
    early_morning = "Early_Morning"
    morning = "Morning"
    afternoon = "Afternoon"
    evening = "Evening"
    night = "Night"


class CabinClass(str, Enum):
    economy = "Economy"
    premium_economy = "Premium_Economy"
    business = "Business"


class Stops(str, Enum):
    zero = "zero"
    one = "one"
    two_or_more = "two_or_more"


class Season(str, Enum):
    winter = "Winter"
    summer = "Summer"
    monsoon = "Monsoon"
    post_monsoon = "Post_Monsoon"


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class FlightPredictionRequest(BaseModel):
    airline: str = Field(..., description="Airline name", example="IndiGo")
    source_city: str = Field(..., description="Source city", example="Delhi")
    destination_city: str = Field(..., description="Destination city", example="Mumbai")
    departure_time: DepartureTime = Field(
        ..., description="Departure time slot", example="Morning"
    )
    cabin_class: CabinClass = Field(..., description="Cabin class", example="Economy")
    days_before_departure: int = Field(
        ..., ge=0, le=90, description="Days before flight (0-90)", example=30
    )
    flight_date: Optional[str] = Field(
        None, description="Flight date (YYYY-MM-DD)", example="2025-11-15"
    )
    distance_km: Optional[float] = Field(
        None, ge=100, le=3000, description="Flight distance in km", example=1150
    )
    stops: Optional[Stops] = Field(
        Stops.zero, description="Number of stops", example="zero"
    )
    is_weekend: Optional[int] = Field(
        0, ge=0, le=1, description="Is weekend flight (0 or 1)", example=0
    )
    is_holiday: Optional[int] = Field(
        0, ge=0, le=1, description="Is holiday (0 or 1)", example=0
    )

    @validator("airline")
    def validate_airline(cls, v):
        if v not in VALID_AIRLINES:
            raise ValueError(
                f"Invalid airline. Must be one of: {', '.join(VALID_AIRLINES)}"
            )
        return v

    @validator("source_city", "destination_city")
    def validate_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"Invalid city. Must be one of: {', '.join(VALID_CITIES)}")
        return v

    @validator("source_city", "destination_city")
    def validate_different_cities(cls, v, values):
        if "source_city" in values and v == values["source_city"]:
            raise ValueError("Source and destination cities must be different")
        return v


class PriceBreakdown(BaseModel):
    base_price: float
    distance_factor: float
    booking_urgency_factor: float
    time_premium: float
    class_multiplier: float
    seasonal_factor: float


class FlightPredictionResponse(BaseModel):
    success: bool
    predicted_price: float
    currency: str = "INR"
    flight_details: dict
    booking_insight: str
    price_trend: str
    model_confidence: str
    savings_tip: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    valid_airlines_count: int
    valid_cities_count: int
    model_metrics: dict


class ValidOptionsResponse(BaseModel):
    airlines: List[str]
    cities: List[str]
    departure_times: List[str]
    cabin_classes: List[str]
    stops_options: List[str]
    seasons: List[str]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def calculate_distance(source: str, destination: str) -> float:
    """Estimate distance between cities (simplified)"""
    # This is a simplified estimation - you could use actual geocoding
    distance_estimates = {
        ("Delhi", "Mumbai"): 1150,
        ("Delhi", "Bangalore"): 1760,
        ("Delhi", "Chennai"): 1760,
        ("Delhi", "Kolkata"): 1320,
        ("Mumbai", "Bangalore"): 850,
        ("Mumbai", "Goa"): 440,
        ("Bangalore", "Goa"): 520,
        ("Delhi", "Goa"): 1420,
        ("Chennai", "Bangalore"): 290,
        ("Mumbai", "Chennai"): 1030,
    }

    key = (source, destination)
    reverse_key = (destination, source)

    if key in distance_estimates:
        return distance_estimates[key]
    elif reverse_key in distance_estimates:
        return distance_estimates[reverse_key]
    else:
        # Rough estimation based on city tiers
        return 1000.0  # Default medium distance


def get_season_from_date(flight_date: datetime) -> str:
    """Determine season from date"""
    month = flight_date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post_Monsoon"


def get_booking_insight(days_before: int, predicted_price: float) -> str:
    """Generate booking insight"""
    if days_before <= 3:
        return f"🔴 Last minute booking! Prices are {predicted_price*0.4:.0f} higher than average."
    elif days_before <= 7:
        return f"🟠 Booking within a week. Consider booking earlier for better prices."
    elif days_before <= 14:
        return f"🟡 Decent booking window. Prices are near average."
    elif days_before <= 30:
        return f"🟢 Good booking time! Prices are competitive."
    elif days_before <= 60:
        return f"🟢 Great! Early booking discount applied."
    else:
        return f"🌟 Excellent! Maximum early bird discount (~15% off)."


def get_price_trend(days_before: int) -> str:
    """Get price trend prediction"""
    if days_before <= 7:
        return "Prices will likely increase as departure approaches"
    elif days_before <= 21:
        return "Prices are relatively stable in this booking window"
    else:
        return "Book now to lock in lower prices before they rise"


def get_savings_tip(
    cabin_class: str, days_before: int, departure_time: str
) -> Optional[str]:
    """Generate savings tip"""
    tips = []

    if days_before < 30:
        tips.append(
            f"💡 Booking 30+ days in advance could save ₹{500 + days_before*50}"
        )

    if cabin_class == "Business" and days_before > 45:
        tips.append("💡 Business class fares may drop 2-3 weeks before departure")

    if departure_time in ["Evening", "Morning"]:
        tips.append("💡 Early morning or night flights are typically 10-15% cheaper")

    return " | ".join(tips) if tips else None


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "✈️ Flight Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "valid_options": "/valid-options",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
        "model_info": {
            "trained_on": f"{metadata['total_flights_trained']:,} flights",
            "rmse": f"₹{metadata['rmse']:.2f}",
            "r2_score": f"{metadata['r2']:.4f}",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "valid_airlines_count": len(VALID_AIRLINES),
        "valid_cities_count": len(VALID_CITIES),
        "model_metrics": {
            "rmse": metadata["rmse"],
            "mae": metadata["mae"],
            "r2": metadata["r2"],
        },
    }


@app.get("/valid-options", response_model=ValidOptionsResponse, tags=["Reference"])
async def get_valid_options():
    """Get all valid options for airlines, cities, etc."""
    return {
        "airlines": sorted(VALID_AIRLINES),
        "cities": sorted(VALID_CITIES),
        "departure_times": [t.value for t in DepartureTime],
        "cabin_classes": [c.value for c in CabinClass],
        "stops_options": [s.value for s in Stops],
        "seasons": [s.value for s in Season],
    }


@app.post("/predict", response_model=FlightPredictionResponse, tags=["Prediction"])
async def predict_price(request: FlightPredictionRequest):
    """
    Predict flight price based on input parameters

    Returns predicted price with detailed insights and booking recommendations.
    """

    try:
        # Parse flight date if provided
        if request.flight_date:
            try:
                flight_dt = datetime.strptime(request.flight_date, "%Y-%m-%d")
                month = flight_dt.month
                season = get_season_from_date(flight_dt)
                day_of_week = flight_dt.strftime("%A")
                is_weekend = 1 if flight_dt.weekday() >= 5 else 0
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
                )
        else:
            month = datetime.now().month
            season = get_season_from_date(datetime.now())
            day_of_week = "Monday"
            is_weekend = request.is_weekend

        # Calculate distance if not provided
        distance_km = request.distance_km or calculate_distance(
            request.source_city, request.destination_city
        )

        # Calculate booking urgency score
        days = request.days_before_departure
        booking_urgency_score = (
            5.0
            if days <= 3
            else (
                3.5
                if days <= 7
                else (
                    2.0
                    if days <= 14
                    else 1.0 if days <= 30 else 0.5 if days <= 60 else 0.2
                )
            )
        )

        # Prepare prediction data
        pred_data = spark.createDataFrame(
            [
                {
                    "airline": request.airline,
                    "source_city": request.source_city,
                    "destination_city": request.destination_city,
                    "departure_time": request.departure_time.value,
                    "class": request.cabin_class.value,
                    "stops": request.stops.value,
                    "distance_km": float(distance_km),
                    "distance_squared": float(distance_km) ** 2,
                    "duration": float(distance_km) / 750.0,
                    "stop_count": (
                        0
                        if request.stops == Stops.zero
                        else (1 if request.stops == Stops.one else 2)
                    ),
                    "days_before_departure": int(days),
                    "booking_urgency_score": booking_urgency_score,
                    "is_weekend": int(is_weekend),
                    "is_holiday": int(request.is_holiday),
                    "month": int(month),
                    "quarter": (int(month) - 1) // 3 + 1,
                    "is_peak_time": (
                        1
                        if request.departure_time.value in ["Morning", "Evening"]
                        else 0
                    ),
                    "route": f"{request.source_city}-{request.destination_city}",
                    "route_frequency": 100,
                    "route_popularity_score": 0.5,
                    "season": season,
                    "day_of_week": day_of_week,
                    "price": 0.0,
                }
            ]
        )

        # Make prediction
        prediction = model.transform(pred_data)
        predicted_price = prediction.select("predicted_price").collect()[0][0]

        # Round to nearest 10
        predicted_price = round(predicted_price / 10) * 10

        # Generate insights
        booking_insight = get_booking_insight(days, predicted_price)
        price_trend = get_price_trend(days)
        savings_tip = get_savings_tip(
            request.cabin_class.value, days, request.departure_time.value
        )

        # Determine confidence based on model metrics
        if metadata["r2"] > 0.9:
            confidence = "Very High (R² > 0.9)"
        elif metadata["r2"] > 0.8:
            confidence = "High (R² > 0.8)"
        else:
            confidence = "Moderate"

        response = {
            "success": True,
            "predicted_price": predicted_price,
            "currency": "INR",
            "flight_details": {
                "route": f"{request.source_city} → {request.destination_city}",
                "airline": request.airline,
                "class": request.cabin_class.value,
                "departure_time": request.departure_time.value,
                "distance_km": distance_km,
                "stops": request.stops.value,
                "days_before_departure": days,
                "flight_date": request.flight_date or "Not specified",
                "is_weekend": bool(is_weekend),
                "is_holiday": bool(request.is_holiday),
            },
            "booking_insight": booking_insight,
            "price_trend": price_trend,
            "model_confidence": confidence,
            "savings_tip": savings_tip,
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predict-quick", tags=["Prediction"])
async def predict_price_quick(
    airline: str = Query(..., description="Airline name"),
    source: str = Query(..., description="Source city"),
    destination: str = Query(..., description="Destination city"),
    cabin_class: str = Query("Economy", description="Cabin class"),
    days_before: int = Query(30, ge=0, le=90, description="Days before departure"),
    departure_time: str = Query("Morning", description="Departure time slot"),
):
    """
    Quick prediction endpoint using GET request with query parameters

    Simplified version for easy testing.
    """

    # Validate inputs
    if airline not in VALID_AIRLINES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid airline. Valid: {', '.join(VALID_AIRLINES)}",
        )

    if source not in VALID_CITIES or destination not in VALID_CITIES:
        raise HTTPException(
            status_code=400, detail=f"Invalid city. Valid: {', '.join(VALID_CITIES)}"
        )

    # Create request object
    try:
        request = FlightPredictionRequest(
            airline=airline,
            source_city=source,
            destination_city=destination,
            departure_time=DepartureTime(departure_time),
            cabin_class=CabinClass(cabin_class),
            days_before_departure=days_before,
        )
        return await predict_price(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Starting Flight Price Prediction API Server")
    print("=" * 80)
    print(f"📍 Server: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🔍 ReDoc: http://localhost:8000/redoc")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
