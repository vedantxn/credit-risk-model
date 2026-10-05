from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import APPROVAL_CUTOFF, load_metadata, load_model, missing_required, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()          # load once at startup, not on the first request
    load_metadata()
    yield


app = FastAPI(title="Credit Risk API", lifespan=lifespan)


class Application(BaseModel):
    """A loan application with the same columns as application_train.csv (without TARGET).

    The typed fields below get range checks. Every column listed in GET /model-info under
    required_input_columns must also be present; other columns are optional.
    """
    model_config = ConfigDict(extra="allow")

    SK_ID_CURR: Optional[int] = None
    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: float = Field(gt=0)
    DAYS_BIRTH: int = Field(lt=0)
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    meta = load_metadata()
    return {
        "model": meta["model"],
        "params": meta["params"],
        "test_metrics": meta["test_metrics"],
        "approval_cutoff": round(APPROVAL_CUTOFF, 4),
        "required_input_columns": meta["required_input_columns"],
    }


@app.post("/predict")
def predict_one(application: Application):
    data = application.model_dump()
    missing = missing_required(data)
    if missing:
        # The model never saw these fields blank in training, so a score would be unreliable.
        raise HTTPException(status_code=422, detail={"error": "missing required fields", "fields": missing})

    result = predict(pd.DataFrame([data])).iloc[0]
    return {
        "SK_ID_CURR": application.SK_ID_CURR,
        "pd_default": round(float(result["pd_default"]), 4),
        "decision": result["decision"],
    }
