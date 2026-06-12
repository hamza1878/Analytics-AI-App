from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
from app.api.routes import overview, demand, anomalies, revenue, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url(
        "redis://localhost:6379",
        decode_responses=True
    )
    print("[api] Redis connected")
    print("[api] Chargement des modèles ML en mémoire…")
    yield
    await app.state.redis.aclose()
    print("[api] Fermeture de l'API ML")


app = FastAPI(
    title="Moviroo ML Intelligence API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(revenue.router, prefix="/revenue")
app.include_router(overview.router, prefix="/intelligence")
app.include_router(demand.router, prefix="/demand")
app.include_router(anomalies.router, prefix="/anomalies")
app.include_router(models.router, prefix="/models")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}