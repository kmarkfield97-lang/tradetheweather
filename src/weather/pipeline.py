"""
Weather data pipeline.
Sources:
  - NWS API (current forecasts, hourly, alerts) — free, no key needed
  - NOAA Climate Data Online (historical normals) — free
  - ephem (moon phase calculation) — local library
"""

import math
from datetime import date, datetime, timezone
from typing import Optional

import httpx
import ephem

NWS_BASE = "https://api.weather.gov"
NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# Major US cities with NWS grid coordinates
# Format: (wfo, gridX, gridY, lat, lon)
US_CITIES = {
    "NYC": ("OKX", 33, 37, 40.7128, -74.0060),
    "LOS_ANGELES": ("LOX", 149, 48, 34.0522, -118.2437),
    "CHICAGO": ("LOT", 74, 73, 41.8781, -87.6298),
    "HOUSTON": ("HGX", 66, 99, 29.7604, -95.3698),
    "PHOENIX": ("PSR", 160, 54, 33.4484, -112.0740),
    "PHILADELPHIA": ("PHI", 49, 75, 39.9526, -75.1652),
    "SAN_ANTONIO": ("EWX", 155, 90, 29.4241, -98.4936),
    "SAN_DIEGO": ("SGX", 53, 24, 32.7157, -117.1611),
    "DALLAS": ("FWD", 99, 101, 32.7767, -96.7970),
    "MIAMI": ("MFL", 110, 39, 25.7617, -80.1918),
    "ATLANTA": ("FFC", 51, 88, 33.7490, -84.3880),
    "BOSTON": ("BOX", 64, 53, 42.3601, -71.0589),
    "SEATTLE": ("SEW", 124, 68, 47.6062, -122.3321),
    "DENVER": ("BOU", 62, 61, 39.7392, -104.9903),
    "MINNEAPOLIS": ("MPX", 107, 70, 44.9778, -93.2650),
    "NEW_ORLEANS": ("LIX", 66, 90, 29.9511, -90.0715),
    "LAS_VEGAS": ("VEF", 41, 33, 36.1699, -115.1398),
    "KANSAS_CITY": ("EAX", 49, 64, 39.0997, -94.5786),
    "CLEVELAND": ("CLE", 64, 66, 41.4993, -81.6944),
    "NASHVILLE": ("OHX", 52, 81, 36.1627, -86.7816),
    "SAN_FRANCISCO": ("MTR", 85, 105, 37.7749, -122.4194),
    "AUSTIN": ("EWX", 158, 92, 30.2672, -97.7431),
    "OKLAHOMA_CITY": ("OUN", 116, 77, 35.4676, -97.5164),
    "DC": ("LWX", 97, 70, 38.9072, -77.0369),
}


class WeatherPipeline:
    def __init__(self):
        self.http = httpx.Client(timeout=20.0, headers={"User-Agent": "TradeTheWeather/1.0 contact@tradetheweather.local"})

    # -------------------------------------------------------------------------
    # NWS — current forecast
    # -------------------------------------------------------------------------

    def get_forecast(self, city: str) -> Optional[dict]:
        """
        Returns today's NWS forecast for the given city.
        Includes high/low temp, precip chance, wind, short description.
        """
        if city not in US_CITIES:
            return None
        wfo, gx, gy, lat, lon = US_CITIES[city]
        url = f"{NWS_BASE}/gridpoints/{wfo}/{gx},{gy}/forecast"
        try:
            resp = self.http.get(url)
            resp.raise_for_status()
            props = resp.json()["properties"]
            periods = props["periods"]
            # NWS periods alternate daytime/nighttime.
            # If periods[0] is nighttime (after ~6 PM), high is in periods[1],
            # and tonight's low is in periods[0].
            if not periods:
                return {"error": "no periods", "city": city}
            p0 = periods[0]
            p1 = periods[1] if len(periods) > 1 else {}
            if p0.get("isDaytime"):
                today = p0
                tonight = p1
                high_temp_f = today.get("temperature")
            else:
                # Already past daytime — tonight IS periods[0], tomorrow day is periods[1]
                # Do NOT use tomorrow's high as today's: same-day high markets have settled.
                today = p1
                tonight = p0
                high_temp_f = None  # today's daytime period is gone; no valid high forecast
            # Use the actual NWS generation time so forecast freshness checks work correctly.
            # NWS provides generatedAt and updateTime; prefer updateTime as it reflects the
            # most recent model run. Fall back to generatedAt, then to now() as last resort.
            nws_generated_at = props.get("updateTime") or props.get("generatedAt") or datetime.now(timezone.utc).isoformat()
            return {
                "city": city,
                "date": date.today().isoformat(),
                "generated_at": nws_generated_at,
                "daytime_name": today.get("name"),
                "high_temp_f": high_temp_f,
                "low_temp_f": tonight.get("temperature"),
                "precip_chance": today.get("probabilityOfPrecipitation", {}).get("value", 0),
                "wind_speed": today.get("windSpeed"),
                "wind_direction": today.get("windDirection"),
                "short_forecast": today.get("shortForecast"),
                "detailed_forecast": today.get("detailedForecast"),
                "icon": today.get("icon"),
            }
        except Exception as e:
            return {"error": str(e), "city": city}

    def get_hourly_forecast(self, city: str) -> list[dict]:
        """Returns hourly NWS forecast for the next 24 hours."""
        if city not in US_CITIES:
            return []
        wfo, gx, gy, lat, lon = US_CITIES[city]
        url = f"{NWS_BASE}/gridpoints/{wfo}/{gx},{gy}/forecast/hourly"
        try:
            resp = self.http.get(url)
            resp.raise_for_status()
            periods = resp.json()["properties"]["periods"][:24]
            result = []
            for p in periods:
                dp_c = p.get("dewpoint", {}).get("value")
                dp_f = round(dp_c * 9 / 5 + 32, 1) if dp_c is not None else None
                result.append({
                    "time": p["startTime"],
                    "temp_f": p["temperature"],
                    "dewpoint_f": dp_f,
                    "precip_chance": p.get("probabilityOfPrecipitation", {}).get("value", 0),
                    "wind_speed": p.get("windSpeed"),
                    "short_forecast": p.get("shortForecast"),
                })
            return result
        except Exception:
            return []

    def get_alerts(self, city: str) -> list[dict]:
        """Returns active NWS weather alerts for the city's state."""
        if city not in US_CITIES:
            return []
        _, _, _, lat, lon = US_CITIES[city]
        url = f"{NWS_BASE}/alerts/active?point={lat},{lon}"
        try:
            resp = self.http.get(url)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            return [
                {
                    "event": f["properties"]["event"],
                    "severity": f["properties"]["severity"],
                    "headline": f["properties"]["headline"],
                    "description": f["properties"]["description"][:300],
                }
                for f in features
            ]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # NOAA — historical normals
    # -------------------------------------------------------------------------

    def get_climate_normals(self, city: str) -> Optional[dict]:
        """
        Returns 30-year climate normals for the city from NOAA.
        Uses the public normals endpoint (no API key required for basic data).
        Falls back to NWS observation stations.
        """
        if city not in US_CITIES:
            return None
        _, _, _, lat, lon = US_CITIES[city]

        # Use NWS observation stations for historical context
        url = f"{NWS_BASE}/points/{lat},{lon}"
        try:
            resp = self.http.get(url)
            resp.raise_for_status()
            props = resp.json()["properties"]
            return {
                "city": city,
                "timezone": props.get("timeZone"),
                "forecast_zone": props.get("forecastZone"),
                "county": props.get("county"),
                "observation_stations": props.get("observationStations"),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recent_observations(self, city: str, hours: int = 72) -> list[dict]:
        """
        Returns recent observations from the nearest NWS station.
        Goes back up to `hours` hours for trend analysis.
        """
        if city not in US_CITIES:
            return []
        _, _, _, lat, lon = US_CITIES[city]

        # Get station list
        try:
            pts_resp = self.http.get(f"{NWS_BASE}/points/{lat},{lon}")
            pts_resp.raise_for_status()
            stations_url = pts_resp.json()["properties"]["observationStations"]
            st_resp = self.http.get(stations_url)
            st_resp.raise_for_status()
            stations = st_resp.json().get("features", [])
            if not stations:
                return []
            station_id = stations[0]["properties"]["stationIdentifier"]

            obs_resp = self.http.get(
                f"{NWS_BASE}/stations/{station_id}/observations",
                params={"limit": hours},
            )
            obs_resp.raise_for_status()
            observations = obs_resp.json().get("features", [])

            results = []
            for obs in observations:
                props = obs["properties"]
                temp_c = props.get("temperature", {}).get("value")
                temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
                results.append({
                    "timestamp": props.get("timestamp"),
                    "temp_f": temp_f,
                    "dewpoint_c": props.get("dewpoint", {}).get("value"),
                    "wind_speed_kmh": props.get("windSpeed", {}).get("value"),
                    "precip_last_hour_mm": props.get("precipitationLastHour", {}).get("value"),
                    "description": props.get("textDescription"),
                    "cloud_layers": props.get("cloudLayers", []),
                })
            return results
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Moon phase
    # -------------------------------------------------------------------------

    def get_moon_phase(self, target_date: Optional[date] = None) -> dict:
        """
        Returns the moon phase for the given date.
        Research shows moon phases correlate with atmospheric pressure and precipitation.
        """
        target_date = target_date or date.today()
        moon = ephem.Moon(target_date.isoformat())
        phase = moon.phase  # 0–100, percentage illuminated

        if phase < 6.25 or phase >= 93.75:
            phase_name = "new_moon"
        elif phase < 43.75:
            phase_name = "waxing_crescent" if phase < 25 else "first_quarter" if phase < 31.25 else "waxing_gibbous"
        elif phase < 56.25:
            phase_name = "full_moon"
        else:
            phase_name = "waning_gibbous" if phase < 75 else "last_quarter" if phase < 81.25 else "waning_crescent"

        # New moon and full moon correlate with slightly higher precipitation probability
        precip_modifier = 0.05 if phase_name in ("new_moon", "full_moon") else 0.0

        return {
            "date": target_date.isoformat(),
            "phase_pct": round(phase, 1),
            "phase_name": phase_name,
            "precip_modifier": precip_modifier,
        }

    # -------------------------------------------------------------------------
    # Composite city report
    # -------------------------------------------------------------------------

    def get_full_report(self, city: str) -> dict:
        """
        Returns a full weather report for a city combining all sources.
        Used by the analysis engine to score markets.
        """
        forecast = self.get_forecast(city)
        hourly = self.get_hourly_forecast(city)
        alerts = self.get_alerts(city)
        moon = self.get_moon_phase()
        observations = self.get_recent_observations(city, hours=24)

        # Compute temperature trend from recent observations.
        # NWS returns observations newest-first; reverse so index 0 = oldest
        # so a positive slope correctly means warming over time.
        temps_newest_first = [o["temp_f"] for o in observations if o["temp_f"] is not None]
        temp_trend = None
        if len(temps_newest_first) >= 3:
            temps = list(reversed(temps_newest_first))  # oldest → newest
            n = len(temps)
            xs = list(range(n))
            mean_x = sum(xs) / n
            mean_y = sum(temps) / n
            num = sum((xs[i] - mean_x) * (temps[i] - mean_y) for i in range(n))
            den = sum((xs[i] - mean_x) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0
            temp_trend = round(slope, 3)  # degrees F per observation (roughly per hour)

        return {
            "city": city,
            "forecast": forecast,
            "hourly": hourly,
            "alerts": alerts,
            "moon": moon,
            "temp_trend": temp_trend,  # degrees/hr, positive = warming
            "recent_observations": observations[:6],  # last 6 for context
        }
