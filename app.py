
import os
import sys
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Code Interpreter with AI Error Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: list[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


def execute_python_code(code: str) -> dict:
    """Execute code in a separate Python process and capture its output."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            script_path.write_text(code, encoding="utf-8")

            process = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=temp_dir,
            )

            output = process.stdout + process.stderr

            return {
                "success": process.returncode == 0,
                "output": output,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "Execution timed out after 5 seconds.",
        }
    except Exception as exc:
        return {
            "success": False,
            "output": f"Execution error: {exc}",
        }


def analyze_error_with_ai(code: str, traceback_text: str) -> list[int]:
    """Ask Gemini to identify the source-code line numbers responsible."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Identify the source-code line number(s) responsible for this
Python execution error.

Use the traceback as the primary evidence. Line numbers must
refer to the supplied CODE, not lines in the traceback.
Return an empty list only if no source line can be identified.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis,
            temperature=0,
        ),
    )

    analysis = ErrorAnalysis.model_validate_json(response.text)

    # Keep only valid, unique source-code line numbers.
    line_count = len(code.splitlines())
    return sorted({
        line for line in analysis.error_lines
        if 1 <= line <= line_count
    })


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception:
        # Do not invent line numbers if AI analysis fails.
        error_lines = []

    return {
        "error": error_lines,
        "result": execution["output"],
    }
