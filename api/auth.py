from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import user_db

router = APIRouter()

class SignupData(BaseModel):
    fullname: str
    gender: str
    contact: str
    email: str
    password: str

class LoginData(BaseModel):
    email: str
    password: str

@router.post("/signup")
def signup(data: SignupData):
    try:
        user_db.create_user(data.fullname, data.gender, data.contact, data.email, data.password)
        return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="User already exists or DB error")

@router.post("/login")
def login(data: LoginData):
    if user_db.verify_user(data.email, data.password):
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
