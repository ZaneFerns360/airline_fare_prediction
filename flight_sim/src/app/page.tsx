"use client";

import React, { useEffect, useState } from "react";
import { Plane, Calendar, MapPin, Clock, TrendingUp, Sparkles, DollarSign, Info, Cloud, Navigation, Users, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { motion } from "framer-motion";

const API_BASE = "http://localhost:8000";

export default function FlightPricePredictor() {
  const [airlines, setAirlines] = useState<string[]>([]);
  const [cities, setCities] = useState<string[]>([]);
  const [departureTimes, setDepartureTimes] = useState<string[]>([]);
  const [cabinClasses, setCabinClasses] = useState<string[]>([]);
  const [stopsOptions, setStopsOptions] = useState<string[]>([]);
  const [seasons, setSeasons] = useState<string[]>([]);

  const [form, setForm] = useState({
    airline: "",
    source_city: "",
    destination_city: "",
    departure_time: "",
    cabin_class: "",
    days_before_departure: 30,
    flight_date: "",
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Fetch dropdown options
  useEffect(() => {
    fetch(`${API_BASE}/valid-options`)
      .then((res) => res.json())
      .then((data) => {
        setAirlines(data.airlines);
        setCities(data.cities);
        setDepartureTimes(data.departure_times);
        setCabinClasses(data.cabin_classes);
        setStopsOptions(data.stops_options);
        setSeasons(data.seasons);
      })
      .catch((err) => console.error("Error fetching valid options:", err));
  }, []);

  const handleChange = (key: string, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const formatLabel = (s: string) => s.replace(/_/g, " ");

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4 md:p-8 flex flex-col items-center">
      {/* Background decorative elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/4 w-60 h-60 bg-cyan-500/5 rounded-full blur-3xl"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-4xl"
      >
        {/* Header */}
        <div className="text-center mb-8 md:mb-12">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="flex justify-center items-center gap-3 mb-4"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-md"></div>
              <Plane className="relative text-blue-400 w-12 h-12 md:w-16 md:h-16 transform rotate-45" />
            </div>
            <h1 className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              Flight Price Predictor
            </h1>
          </motion.div>
          <p className="text-blue-200/80 text-lg md:text-xl max-w-2xl mx-auto">
            Smart predictions for your next journey. Find the best time to book and save money.
          </p>
        </div>

        <Card className="w-full bg-slate-800/60 backdrop-blur-md border-blue-500/20 shadow-2xl rounded-2xl overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-cyan-400"></div>

          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-3 text-2xl md:text-3xl font-bold text-white">
              <Navigation className="text-blue-400 w-7 h-7" />
              Plan Your Flight
            </CardTitle>
            <p className="text-blue-200/70 text-base md:text-lg mt-2">
              Fill in your travel details to get an accurate price prediction
            </p>
          </CardHeader>

          <CardContent className="space-y-6 pt-2">
            {/* FORM */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 md:gap-6">
              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <Users className="w-4 h-4 text-blue-300" />
                  Airline
                </Label>
                <Select onValueChange={(v) => handleChange("airline", v)}>
                  <SelectTrigger className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base hover:bg-slate-700/70 transition-colors">
                    <SelectValue placeholder="Select Airline" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-blue-400/30 text-white">
                    {airlines.map((a) => (
                      <SelectItem key={a} value={a} className="text-base focus:bg-blue-600/50">
                        {formatLabel(a)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <Shield className="w-4 h-4 text-blue-300" />
                  Cabin Class
                </Label>
                <Select onValueChange={(v) => handleChange("cabin_class", v)}>
                  <SelectTrigger className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base hover:bg-slate-700/70 transition-colors">
                    <SelectValue placeholder="Select Cabin Class" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-blue-400/30 text-white">
                    {cabinClasses.map((c) => (
                      <SelectItem key={c} value={c} className="text-base focus:bg-blue-600/50">
                        {formatLabel(c)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-blue-300" />
                  Source City
                </Label>
                <Select onValueChange={(v) => handleChange("source_city", v)}>
                  <SelectTrigger className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base hover:bg-slate-700/70 transition-colors">
                    <SelectValue placeholder="Select Source" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-blue-400/30 text-white">
                    {cities.map((c) => (
                      <SelectItem key={c} value={c} className="text-base focus:bg-blue-600/50">
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-blue-300" />
                  Destination City
                </Label>
                <Select onValueChange={(v) => handleChange("destination_city", v)}>
                  <SelectTrigger className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base hover:bg-slate-700/70 transition-colors">
                    <SelectValue placeholder="Select Destination" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-blue-400/30 text-white">
                    {cities.map((c) => (
                      <SelectItem key={c} value={c} className="text-base focus:bg-blue-600/50">
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-300" />
                  Departure Time
                </Label>
                <Select onValueChange={(v) => handleChange("departure_time", v)}>
                  <SelectTrigger className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base hover:bg-slate-700/70 transition-colors">
                    <SelectValue placeholder="Select Time" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-blue-400/30 text-white">
                    {departureTimes.map((t) => (
                      <SelectItem key={t} value={t} className="text-base focus:bg-blue-600/50">
                        {formatLabel(t)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-blue-300" />
                  Days Before Departure
                </Label>
                <Input
                  type="number"
                  min={0}
                  value={form.days_before_departure}
                  onChange={(e) => handleChange("days_before_departure", +e.target.value)}
                  className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="space-y-3 md:col-span-2">
                <Label className="text-blue-100 font-semibold text-base flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-blue-300" />
                  Flight Date
                </Label>
                <Input
                  type="date"
                  value={form.flight_date}
                  onChange={(e) => handleChange("flight_date", e.target.value)}
                  className="bg-slate-700/50 border-blue-400/30 text-white h-12 text-base focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold py-3 text-lg h-14 mt-2 shadow-lg shadow-blue-500/30 transition-all duration-300"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Predicting Your Flight Price...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    Predict Flight Price
                  </div>
                )}
              </Button>
            </motion.div>

            {/* RESULT */}
            {result && result.success && (
              <motion.div
                className="mt-6 p-6 rounded-xl bg-gradient-to-br from-slate-800/80 to-blue-900/40 border border-blue-500/30 shadow-2xl space-y-4 backdrop-blur-sm"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="flex items-start justify-between">
                  <h2 className="text-2xl font-bold flex items-center gap-3 text-white">
                    <div className="p-2 bg-green-500/20 rounded-lg">
                      <DollarSign className="text-green-400 w-6 h-6" />
                    </div>
                    Predicted Price: <span className="text-green-400">₹{result.predicted_price.toLocaleString("en-IN")}</span>
                  </h2>
                  <div className="px-3 py-1 bg-blue-500/20 rounded-full border border-blue-400/30">
                    <span className="text-blue-300 text-sm font-medium">AI Powered</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  <div className="space-y-4">
                    <div className="flex items-start gap-3 p-3 bg-blue-500/10 rounded-lg border border-blue-400/20">
                      <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                      <p className="text-blue-100 text-base">{result.booking_insight}</p>
                    </div>

                    <div className="flex items-start gap-3 p-3 bg-amber-500/10 rounded-lg border border-amber-400/20">
                      <TrendingUp className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
                      <p className="text-amber-100 text-base">{result.price_trend}</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-start gap-3 p-3 bg-purple-500/10 rounded-lg border border-purple-400/20">
                      <Sparkles className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
                      <p className="text-purple-100 text-base">Confidence: {result.model_confidence}</p>
                    </div>

                    <div className="flex items-start gap-3 p-3 bg-green-500/10 rounded-lg border border-green-400/20">
                      <DollarSign className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                      <p className="text-green-100 text-base">{result.savings_tip}</p>
                    </div>
                  </div>
                </div>

                <div className="border-t border-blue-400/20 pt-5 mt-4">
                  <h3 className="text-xl font-bold flex items-center gap-3 text-white mb-4">
                    <div className="p-2 bg-blue-500/20 rounded-lg">
                      <MapPin className="text-blue-400 w-5 h-5" />
                    </div>
                    Flight Details
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Route</span>
                        <span className="text-white font-semibold">{result.flight_details.route}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Airline</span>
                        <span className="text-white font-semibold">{result.flight_details.airline}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Departure</span>
                        <span className="text-white font-semibold">{result.flight_details.departure_time}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Class</span>
                        <span className="text-white font-semibold">{result.flight_details.class}</span>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Date</span>
                        <span className="text-white font-semibold">{result.flight_details.flight_date}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Distance</span>
                        <span className="text-white font-semibold">{result.flight_details.distance_km} km</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Stops</span>
                        <span className="text-white font-semibold">{result.flight_details.stops}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-blue-400/10">
                        <span className="text-blue-200 font-medium">Days Before</span>
                        <span className="text-white font-semibold">{result.flight_details.days_before_departure}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </CardContent>
        </Card>

        {/* Footer note */}
        <div className="text-center mt-8">
          <p className="text-blue-300/60 text-sm">
            Powered by advanced machine learning • Real-time price predictions • Travel smarter
          </p>
        </div>
      </motion.div>
    </div>
  );
}

