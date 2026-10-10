
import re


def get_error_lines_from_traceback(traceback_text: str) -> list[int]:
    """Extract source line numbers from the executed script's traceback."""
    matches = re.findall(
        r'File "([^"]*script\.py)", line (\d+)',
        traceback_text
    )
    return sorted({int(line) for _, line in matches})


def analyze_error_with_ai(code: str, traceback_text: str) -> list[int]:
    """Use Gemini to analyze the error, with traceback-based fallback."""
    ai_lines = []

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)

            prompt = f"""
Identify the source line responsible for the Python error.
Return JSON with an error_lines array of integers.
Use line numbers from the supplied code, starting at 1.

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
            line_count = len(code.splitlines())
            ai_lines = sorted({
                line for line in analysis.error_lines
                if 1 <= line <= line_count
            })

    except Exception:
        pass

    # Prefer the executed script's actual traceback line numbers.
    traceback_lines = get_error_lines_from_traceback(traceback_text)

    return traceback_lines if traceback_lines else ai_lines


def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"],
    )

    return {
        "error": error_lines,
        "result": execution["output"],
    }
