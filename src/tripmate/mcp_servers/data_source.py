"""Configurable mock and free-API data sources for TripMate MCP tools."""

from datetime import date as Date
from typing import Any, Protocol

import httpx

from tripmate.config import Settings


class TravelDataSource(Protocol):
    """Data-provider contract implemented by MCP server backends."""

    def search_flights(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]:
        """Return flight options for a route and travel date."""

    def search_hotels(
        self, city: str, checkin: str, checkout: str, max_price: float | None
    ) -> list[dict[str, Any]]:
        """Return hotel options for a stay and price ceiling."""

    async def get_weather(self, city: str, date: str) -> dict[str, Any]:
        """Return weather information for a city and date."""

    def search_places(self, city: str, category: str) -> list[dict[str, Any]]:
        """Return points of interest for a city and category."""


class MockTravelDataSource:
    """Deterministic sample provider for all TripMate tools."""

    def search_flights(
        self, origin: str, destination: str, date: str
    ) -> list[dict[str, Any]]:
        """Return sample options without contacting an airline API."""
        return [
            {
                "option_id": "flight-1",
                "airline": "IndiGo",
                "flight_number": "6E 512",
                "departure": f"{origin} to {destination}, {date}, morning",
                "arrival": "Nonstop",
                "price_inr": 9500,
            },
            {
                "option_id": "flight-2",
                "airline": "Air India",
                "flight_number": "AI 865",
                "departure": f"{origin} to {destination}, {date}, afternoon",
                "arrival": "Nonstop",
                "price_inr": 12500,
            },
            {
                "option_id": "flight-3",
                "airline": "IndiGo",
                "flight_number": "6E 204 / 6E 702",
                "departure": f"{origin} to {destination}, {date}, evening",
                "arrival": "One stop",
                "price_inr": 8000,
            },
        ]

    def search_hotels(
        self, city: str, checkin: str, checkout: str, max_price: float | None
    ) -> list[dict[str, Any]]:
        """Return sample stays filtered by the estimated total price."""
        try:
            nights = max((Date.fromisoformat(checkout) - Date.fromisoformat(checkin)).days, 1)
        except ValueError:
            nights = 4
        options = [
            {
                "option_id": "hotel-1",
                "name": "Palm Grove Guesthouse",
                "area": f"Calangute, {city}",
                "nightly_rate_inr": 2500,
                "total_cost_inr": 2500 * nights,
                "rating": 4.0,
            },
            {
                "option_id": "hotel-2",
                "name": "Coastal Courtyard",
                "area": f"Candolim, {city}",
                "nightly_rate_inr": 3800,
                "total_cost_inr": 3800 * nights,
                "rating": 4.4,
            },
            {
                "option_id": "hotel-3",
                "name": "The Seabreeze Resort",
                "area": f"Baga, {city}",
                "nightly_rate_inr": 6200,
                "total_cost_inr": 6200 * nights,
                "rating": 4.7,
            },
        ]
        affordable = [
            option
            for option in options
            if max_price is None or option["total_cost_inr"] <= max_price
        ]
        return affordable or options[:1]

    async def get_weather(self, city: str, date: str) -> dict[str, Any]:
        """Return clearly labeled sample weather."""
        return {
            "source": "mock",
            "city": city,
            "date": date,
            "condition": "Partly cloudy",
            "temperature_c": 28,
            "note": "Sample weather only; verify before travel.",
        }

    def search_places(self, city: str, category: str) -> list[dict[str, Any]]:
        """Return sample points of interest in the requested category."""
        places = [
            ("Fort Aguada", "history", "Historic fort and coastal views."),
            ("Baga Beach", "beach", "Beach with cafes and water activities."),
            ("Dudhsagar Falls", "nature", "Waterfall and forest scenery."),
            ("Anjuna Market", "shopping", "Local market for crafts and food."),
        ]
        normalized = category.casefold()
        return [
            {"name": name, "category": kind, "city": city, "description": description}
            for name, kind, description in places
            if normalized in {"any", "all", "sightseeing"} or normalized in kind
        ]


class OpenMeteoTravelDataSource(MockTravelDataSource):
    """Free Open-Meteo weather with mock flight, hotel, and place results."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    async def get_weather(self, city: str, date: str) -> dict[str, Any]:
        """Resolve a city and query the no-key Open-Meteo forecast API."""
        try:
            Date.fromisoformat(date)
        except ValueError:
            return {
                "source": "open-meteo",
                "city": city,
                "date": date,
                "error": "Use a calendar date in YYYY-MM-DD format for forecast data.",
            }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            geocode_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )
            geocode_response.raise_for_status()
            results = geocode_response.json().get("results", [])
            if not results:
                return {
                    "source": "open-meteo",
                    "city": city,
                    "date": date,
                    "error": "City was not found by Open-Meteo geocoding.",
                }
            place = results[0]
            forecast_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "start_date": date,
                    "end_date": date,
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            daily = forecast_response.json().get("daily", {})
            if not daily.get("time"):
                return {
                    "source": "open-meteo",
                    "city": city,
                    "date": date,
                    "error": "No forecast is available for that date.",
                }
            return {
                "source": "open-meteo",
                "city": place.get("name", city),
                "date": date,
                "weather_code": daily["weather_code"][0],
                "temperature_max_c": daily["temperature_2m_max"][0],
                "temperature_min_c": daily["temperature_2m_min"][0],
                "precipitation_probability_percent": daily[
                    "precipitation_probability_max"
                ][0],
            }


def create_data_source(settings: Settings) -> TravelDataSource:
    """Select the configured data provider."""
    if settings.tripmate_data_source == "open-meteo":
        return OpenMeteoTravelDataSource(settings.tripmate_mcp_tool_timeout_seconds)
    return MockTravelDataSource()