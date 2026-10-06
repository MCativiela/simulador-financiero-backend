import os

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fintech Analytics API")

NOMBRES_ACTIVOS = {
    "AAPL": "Apple Inc.",
    "AAPL3": "Apple Inc.",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla Inc.",
}


def obtener_nombre_activo(ticker: str) -> str:
    return NOMBRES_ACTIVOS.get(ticker, ticker)


origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "mensaje": "API Fintech ejecutándose correctamente",
    }


@app.get("/api/analisis/healthz")
def api_health():
    return {"status": "ok"}


@app.get("/api/analisis/buscar")
def buscar_activos(q: str = Query(..., min_length=2, max_length=80)):
    consulta = q.strip()
    if len(consulta) < 2:
        raise HTTPException(
            status_code=400,
            detail="Ingresá al menos dos caracteres para buscar un activo.",
        )

    tipos_admitidos = {
        "EQUITY",
        "ETF",
        "CRYPTOCURRENCY",
        "CURRENCY",
        "INDEX",
        "MUTUALFUND",
    }

    try:
        respuesta = yf.Search(consulta, max_results=15)
        resultados = []
        simbolos_vistos = set()

        for activo in respuesta.quotes or []:
            simbolo = activo.get("symbol")
            tipo = activo.get("quoteType", "")
            if not simbolo or tipo not in tipos_admitidos or simbolo in simbolos_vistos:
                continue

            simbolos_vistos.add(simbolo)
            resultados.append(
                {
                    "symbol": simbolo,
                    "name": activo.get("shortname") or activo.get("longname") or simbolo,
                    "exchange": activo.get("exchDisp") or activo.get("exchange") or "",
                    "type": activo.get("typeDisp") or tipo,
                }
            )

            if len(resultados) == 8:
                break

        return {"results": resultados}
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="No se pudo buscar activos en Yahoo Finance.",
        ) from error


@app.get("/api/analisis/{ticker}")
def analizar_activo(
    ticker: str,
    fecha_desde: str = Query(..., examples=["2023-01-01"]),
    fecha_hasta: str = Query(..., examples=["2026-09-05"]),
):
    try:
        ticker_normalizado = ticker.strip().upper()
        if not ticker_normalizado:
            raise HTTPException(status_code=400, detail="El ticker es obligatorio.")
        if fecha_desde >= fecha_hasta:
            raise HTTPException(
                status_code=400,
                detail="La fecha desde debe ser anterior a la fecha hasta.",
            )

        df = yf.download(
            ticker_normalizado,
            start=fecha_desde,
            end=fecha_hasta,
            progress=False,
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos para el ticker o rango especificado.",
            )

        if "Close" in df.columns:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato de respuesta inesperado de Yahoo Finance.",
            )

        close = close.dropna()
        if len(close) < 2:
            raise HTTPException(
                status_code=400,
                detail="No hay suficientes datos para realizar el cálculo financiero.",
            )

        precio_inicial = float(close.iloc[0])
        precio_final = float(close.iloc[-1])
        retorno_total = ((precio_final - precio_inicial) / precio_inicial) * 100

        retornos_diarios = close.pct_change().dropna()
        volatilidad_anual = retornos_diarios.std() * np.sqrt(252) * 100
        sma_50_series = close.rolling(window=50).mean().fillna(0)

        labels = [idx.strftime("%b %Y") for idx in close.index]
        precios = [round(float(value), 2) for value in close.values]
        sma_50 = [round(float(value), 2) for value in sma_50_series.values]

        return {
            "status": "success",
            "ticker": ticker_normalizado,
            "name": obtener_nombre_activo(ticker_normalizado),
            "finalPrice": f"${round(precio_final, 2)}",
            "returnPercentage": round(float(retorno_total), 2),
            "volatility": f"{round(float(volatilidad_anual), 1)}%",
            "chartData": {
                "labels": labels,
                "prices": precios,
                "sma_50": sma_50,
            },
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error