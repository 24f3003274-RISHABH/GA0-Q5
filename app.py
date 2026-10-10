
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="Code Interpreter with AI Error Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request and response models
# --------------------------------------------------

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: list[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


# --------------------------------------------------
# Tool: Execute Python code
# --------------------------------------------------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code in a separate process.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """
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

            # Preserve stdout and stderr exactly as captured.
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


# --------------------------------------------------
# Extract line numbers from the traceback
# --------------------------------------------------

def get_error_lines_from_traceback(
    traceback_text: str
) -> list[int]:
    """
    Extract source line numbers from the executed script's traceback.
    """
    matches = re.findall(
        r'File ".*?script\.py", line (\d+)',
        traceback_text
    )

    return sorted({int(line) for line in matches})


# --------------------------------------------------
# AI: Analyze execution errors using Gemini
# --------------------------------------------------

def analyze_error_with_ai(
    code: str,
    traceback_text: str
) -> list[int]:
    """
    Ask Gemini to identify the source-code error lines.
    Use traceback line numbers as a reliable fallback.
    """
    ai_lines = []

    try:
        api_key = os.environ.get("GEMINI_API_KEY")

        if api_key:
            client = genai.Client(api_key=api_key)

            prompt = f"""
Analyze the Python code and its execution traceback.

Identify the source-code line number or numbers responsible
for the error. Count code lines starting from 1.

Return only structured JSON matching the requested schema.

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

            analysis = ErrorAnalysis.model_validate_json(
                response.text
            )

            line_count = len(code.splitlines())

            ai_lines = sorted({
                line
                for line in analysis.error_lines
                if 1 <= line <= line_count
            })

    except Exception:
        # Continue to traceback-based analysis if Gemini fails.
        ai_lines = []

    # Prefer line numbers from the actual Python traceback.
    traceback_lines = get_error_lines_from_traceback(
        traceback_text
    )

    if traceback_lines:
        return traceback_lines

    return ai_lines


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Code Interpreter API is running"
    }


# --------------------------------------------------
# Main endpoint: POST /code-interpreter
# --------------------------------------------------

@app.post(
    "/code-interpreter",
    response_model=CodeResponse
)
def code_interpreter(request: CodeRequest):
    # Step 1: Execute the submitted code.
    execution = execute_python_code(request.code)

    # Step 2: Successful execution does not need AI.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # Step 3: Analyze errors only when execution fails.
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"],
    )

    # Step 4: Return error lines and execution output.
    return {
        "error": error_lines,
        "result": execution["output"],
    }
