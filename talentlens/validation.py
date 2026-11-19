"""A deliberately small JSON-schema-style validator.

Supports the subset of JSON Schema the API needs: ``type``, ``properties``,
``required``, ``additionalProperties: false``, ``minLength``/``maxLength``,
``minItems``/``maxItems``, ``items``, ``minimum``/``maximum`` and ``enum``.
Returning a list of human-readable problems (rather than raising on the
first) lets the API report every issue at once.
"""

from __future__ import annotations

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}


def validate(instance, schema: dict, path: str = "$") -> list[dict]:
    """Return a list of ``{"field": path, "message": ...}`` problems (empty when valid)."""
    problems: list[dict] = []
    expected = schema.get("type")
    if expected:
        python_type = _TYPES[expected]
        ok = isinstance(instance, python_type)
        if expected in {"integer", "number"} and isinstance(instance, bool):
            ok = False
        if not ok:
            return [{"field": path, "message": f"must be of type {expected}"}]

    if "enum" in schema and instance not in schema["enum"]:
        problems.append({"field": path, "message": "must be one of " + ", ".join(map(str, schema["enum"]))})

    if isinstance(instance, str):
        if "minLength" in schema and len(instance.strip()) < schema["minLength"]:
            problems.append({"field": path, "message": f"must be at least {schema['minLength']} characters"})
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            problems.append({"field": path, "message": f"must be at most {schema['maxLength']} characters"})

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            problems.append({"field": path, "message": f"must be >= {schema['minimum']}"})
        if "maximum" in schema and instance > schema["maximum"]:
            problems.append({"field": path, "message": f"must be <= {schema['maximum']}"})

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            problems.append({"field": path, "message": f"must contain at least {schema['minItems']} item(s)"})
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            problems.append({"field": path, "message": f"must contain at most {schema['maxItems']} item(s)"})
        if "items" in schema:
            for index, item in enumerate(instance):
                problems.extend(validate(item, schema["items"], f"{path}[{index}]"))

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                problems.append({"field": f"{path}.{name}", "message": "is required"})
        if schema.get("additionalProperties") is False:
            for name in instance:
                if name not in properties:
                    problems.append({"field": f"{path}.{name}", "message": "is not an allowed field"})
        for name, subschema in properties.items():
            if name in instance:
                problems.extend(validate(instance[name], subschema, f"{path}.{name}"))
    return problems


JOB_DESCRIPTION = {"type": "string", "minLength": 20, "maxLength": 20000}
JOB_TITLE = {"type": "string", "maxLength": 120}
MUST_HAVE = {"type": "array", "maxItems": 20, "items": {"type": "string", "minLength": 1, "maxLength": 60}}

SCORE_REQUEST = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_description", "resumes"],
    "properties": {
        "job_title": JOB_TITLE,
        "job_description": JOB_DESCRIPTION,
        "must_have": MUST_HAVE,
        "resumes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "text"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 120},
                    "text": {"type": "string", "maxLength": 100000},
                },
            },
        },
    },
}

SCREEN_FORM = {
    "type": "object",
    "required": ["job_description"],
    "properties": {"job_title": JOB_TITLE, "job_description": JOB_DESCRIPTION, "must_have": MUST_HAVE},
}

CANDIDATE_PATCH = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "shortlisted": {"type": "boolean"},
        "notes": {"type": "string", "maxLength": 5000},
    },
}
