"""The nine booking inputs shared by training, HTML forms and CSV prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FIELDS = {
    "lead_time": ("Antelación (días)", 0, 3650, True, 85),
    "no_of_special_requests": ("Solicitudes especiales", 0, 20, True, 0),
    "avg_price_per_room": ("Precio por noche (€)", 0, 100000, False, 103.42),
    "no_of_adults": ("Adultos", 0, 20, True, 2),
    "no_of_weekend_nights": ("Noches de fin de semana", 0, 365, True, 1),
    "required_car_parking_space": ("Plaza de aparcamiento", 0, 1, True, 0),
    "no_of_week_nights": ("Noches entre semana", 0, 365, True, 2),
    "arrival_month": ("Mes de llegada", 1, 12, True, 7),
}
FEATURES = [*NUMERIC_FIELDS, "market_segment_type_Online"]
INPUTS = [*NUMERIC_FIELDS, "market_segment_type"]
LABELS = {key: value[0] for key, value in NUMERIC_FIELDS.items()}
LABELS["market_segment_type_Online"] = "Reserva online"
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


class InputError(ValueError):
    """A problem the user can correct without a traceback."""


def normalize_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate inputs, deriving month from the notebook's ISO arrival date."""
    frame = frame.copy()
    if frame.empty:
        raise InputError("El archivo no contiene reservas.")
    if "arrival_month" not in frame and "arrival_date" in frame:
        dates = pd.to_datetime(frame["arrival_date"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise InputError("arrival_date debe contener fechas válidas con formato AAAA-MM-DD.")
        frame["arrival_month"] = dates.dt.month
    missing = [key for key in INPUTS if key not in frame]
    if missing:
        raise InputError("Faltan columnas: " + ", ".join(missing))
    for key, (label, low, high, integer, _) in NUMERIC_FIELDS.items():
        values = pd.to_numeric(frame[key], errors="coerce")
        invalid = ~np.isfinite(values) | (values < low) | (values > high)
        if integer:
            invalid |= values.mod(1).ne(0)
        if invalid.any():
            row = int(np.flatnonzero(invalid.to_numpy())[0]) + 1
            kind = "un entero" if integer else "un número"
            raise InputError(f"Fila {row}: {label} debe ser {kind} entre {low} y {high}.")
        frame[key] = values.astype(int if integer else float)
    frame["market_segment_type"] = frame["market_segment_type"].astype(str).str.strip()
    if not frame["market_segment_type"].isin(["Online", "Offline"]).all():
        raise InputError("market_segment_type debe ser Online u Offline.")
    return frame


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_inputs(frame)
    frame["market_segment_type_Online"] = frame["market_segment_type"].eq("Online").astype(int)
    return frame[FEATURES]


def load_data(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "booking_status" not in data:
        raise InputError("El CSV de entrenamiento necesita booking_status.")
    if not data["booking_status"].isin(["Canceled", "Not_Canceled"]).all():
        raise InputError("booking_status debe ser Canceled o Not_Canceled.")
    data = normalize_inputs(data)
    if "arrival_date" not in data:
        raise InputError("El conjunto de entrenamiento necesita arrival_date (AAAA-MM-DD).")
    dates = pd.to_datetime(data["arrival_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.month.eq(data["arrival_month"]).all():
        raise InputError("arrival_date debe ser una fecha válida y coincidir con arrival_month.")
    if data["booking_status"].value_counts().min() < 10 or data["booking_status"].nunique() != 2:
        raise InputError("Se necesitan al menos 10 reservas de cada clase para entrenar.")
    data["arrival_date"] = dates.dt.strftime("%Y-%m-%d")
    return data
