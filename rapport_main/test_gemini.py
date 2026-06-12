"""
Run: python test_gemini.py
Shows exactly what Gemini returns and why the parse fails.
"""
import os, json, httpx, asyncio
from dotenv import load_dotenv

load_dotenv(dotenv_path="rapport_main/.env")

KEY = os.environ["GEMINI_API_KEY"]

# Test 1: v1beta with responseMimeType
async def test_v1beta():
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={KEY}"
    body = {
        "contents": [{"parts": [{"text": 'Return a JSON object: {"hello": "world", "status": "ok"}'}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
        },
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, json=body)
    print(f"\n=== v1beta JSON mode ===")
    print(f"HTTP: {r.status_code}")
    if r.is_success:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"RAW: {repr(raw)}")
        try:
            print(f"PARSED: {json.loads(raw)}")
            print("✅ JSON mode WORKS")
        except Exception as e:
            print(f"❌ Parse failed: {e}")
    else:
        print(f"ERROR: {r.text[:500]}")

# Test 2: v1beta without responseMimeType (plain text)
async def test_plain():
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={KEY}"
    body = {
        "contents": [{"parts": [{"text": 'Return ONLY this JSON, no markdown: {"hello": "world"}'}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, json=body)
    print(f"\n=== v1beta plain text ===")
    print(f"HTTP: {r.status_code}")
    if r.is_success:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"RAW: {repr(raw)}")
        try:
            import re
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            print(f"PARSED: {json.loads(clean)}")
            print("✅ Plain text WORKS")
        except Exception as e:
            print(f"❌ Parse failed: {e}")
    else:
        print(f"ERROR: {r.text[:500]}")

# Test 3: check quota / key validity
async def test_key():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={KEY}"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url)
    print(f"\n=== API key check ===")
    print(f"HTTP: {r.status_code}")
    if r.is_success:
        models = [m["name"] for m in r.json().get("models", []) if "gemini" in m["name"]]
        print(f"Available models: {models[:5]}")
        print("✅ Key is valid")
    else:
        print(f"❌ Key error: {r.text[:300]}")

asyncio.run(test_key())
asyncio.run(test_v1beta())
asyncio.run(test_plain())