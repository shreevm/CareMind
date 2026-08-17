import re

from .schemas import RetrievedChunk
from .tools import clean_evidence_text


def clinical_summary_answer(chunks: list[RetrievedChunk]) -> str:
    evidence_texts = [clean_evidence_text(chunk.text) for chunk in chunks]
    combined = "\n".join(evidence_texts)
    all_citations = citation_span(len(chunks))
    patient_profile = extract_patient_profile(combined)
    findings = extract_holter_findings(combined) or extract_clinical_findings(evidence_texts)
    triggered = extract_unique_matches(
        combined,
        [
            r"Patient Triggered[^:\n]*(?::|\n)?\s*[^\n.]{0,160}",
            r"Manual[^:\n]*(?::|\n)?\s*[^\n.]{0,160}",
        ],
    )

    lines = [
        f"From the retrieved evidence, the uploaded material contains the following documented clinical details {all_citations}.",
    ]
    if patient_profile:
        lines.extend(
            [
                "",
                "Patient profile:",
                f"- Name: {patient_profile.get('name', 'Not found')} [1]",
                f"- ID: {patient_profile.get('id', 'Not found')} [1]",
                f"- Age: {patient_profile.get('age', 'Not found')} [1]",
                f"- Gender: {patient_profile.get('gender', 'Not found')} [1]",
            ]
        )
    lines.extend(["", "Key findings:"])
    if findings:
        for finding in findings[:10]:
            lines.append(f"- {finding} {all_citations}")
    else:
        lines.append("- I found rhythm-monitoring entries, but the extracted PDF text does not include a clear final impression section. [1]")

    if triggered:
        lines.append("")
        lines.append("Patient/manual event evidence:")
        for item in triggered[:3]:
            lines.append(f"- {item} [1]")

    lines.extend(
        [
            "",
            "Evidence-grounded synthesis:",
            f"- The summary above is limited to findings explicitly present in the retrieved report text; diagnosis, treatment decisions, and unsupported clinical significance are not provided in the retrieved evidence {all_citations}.",
        ]
    )
    return "\n".join(lines)


def citation_span(chunk_count: int) -> str:
    if chunk_count <= 1:
        return "[1]"
    return "[" + ", ".join(str(index) for index in range(1, chunk_count + 1)) + "]"


def extract_holter_findings(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    lowered = compact.lower()
    if not any(term in lowered for term in ["holter", "ecg", "sinus", "svt", "ectopic", "heart rate", "hr"]):
        return []

    findings: list[str] = []
    max_hr = first_match(compact, [r"\b(?:max|maximum)(?:imum)?(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b"])
    avg_hr = first_match(compact, [r"\b(?:avg|average)(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b"])
    min_hr = first_match(compact, [r"\b(?:min|minimum)(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b"])
    heart_rate_parts = []
    if max_hr:
        heart_rate_parts.append(f"maximum {max_hr}")
    if avg_hr:
        heart_rate_parts.append(f"average {avg_hr}")
    if min_hr:
        heart_rate_parts.append(f"minimum {min_hr}")
    if heart_rate_parts:
        findings.append("Overall heart rate: " + ", ".join(heart_rate_parts) + ".")

    if re.search(r"\bsinus rhythm\b", compact, flags=re.IGNORECASE):
        findings.append("Baseline/recorded rhythm includes sinus rhythm.")

    svt_count = first_match(compact, [r"\b(\d+)\s+episodes?\s+of\s+SVT\b", r"\bSVT\s+episodes?\D{0,40}(\d+)\b"])
    if svt_count:
        svt_detail = first_match(
            compact,
            [
                r"\bSVT\b.{0,80}?(\d+\s*bpm.{0,40}?\d+(?:\.\d+)?\s*(?:secs?|seconds?))",
                r"(\d+\s*bpm.{0,40}?\d+(?:\.\d+)?\s*(?:secs?|seconds?)).{0,40}?\bSVT\b",
            ],
        )
        findings.append(f"SVT noted: {svt_count} episode(s){f' ({svt_detail})' if svt_detail else ''}.")
    elif re.search(r"\bSVT\b|supraventricular tachycardia", compact, flags=re.IGNORECASE):
        findings.append("Supraventricular tachycardia/SVT is mentioned in the report.")

    if re.search(r"\bsinus tachycardia\b", compact, flags=re.IGNORECASE):
        tachy_hr = first_match(compact, [r"\bsinus tachycardia\b.{0,60}?(\d+\s*bpm)"])
        findings.append(f"Sinus tachycardia is noted{f' around {tachy_hr}' if tachy_hr else ''}.")

    ventricular, supraventricular = extract_ectopic_counts(compact)
    ectopic_parts = []
    if ventricular:
        ectopic_parts.append(f"ventricular ectopics {ventricular}")
    if supraventricular:
        ectopic_parts.append(f"supraventricular ectopics {supraventricular}")
    if ectopic_parts:
        findings.append("Ectopic burden/counts: " + "; ".join(ectopic_parts) + ".")
    elif re.search(r"\bectopic", compact, flags=re.IGNORECASE):
        findings.append("Ectopic beats are mentioned; use the source report for exact burden/counts.")

    negatives = []
    for label, pattern in [
        ("VT", r"\bno\s+(?:vt|ventricular tachycardia)\b"),
        ("AF", r"\bno\s+(?:af|atrial fibrillation)\b"),
        ("advanced AV block", r"\bno\s+(?:advanced\s+)?av blocks?\b"),
        ("pauses", r"\bno\s+pauses?\b"),
    ]:
        if re.search(pattern, compact, flags=re.IGNORECASE):
            negatives.append(label)
    if negatives:
        findings.append("No " + ", ".join(negatives) + " reported.")

    if re.search(r"\bno symptoms?\b|\bno patient triggered\b|\bno manual\b", compact, flags=re.IGNORECASE):
        findings.append("No patient symptoms/manual events are documented in the retrieved evidence.")

    return dedupe_findings(findings)


def extract_ectopic_counts(text: str) -> tuple[str, str]:
    table_match = re.search(
        r"\bVentricular\s+Supraventricular\s+Total\s+([0-9,]+)\s+([0-9,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if table_match:
        return table_match.group(1), table_match.group(2)
    ventricular = first_match(
        text,
        [r"\bventricular(?:\s+ectopic(?:s|\s*\(?ve\)?)?)?\s*[:=-]?\s*([0-9,]+)\s*(?:total|beats?)\b"],
    )
    supraventricular = first_match(
        text,
        [r"\bsupraventricular(?:\s+ectopic(?:s|\s*\(?sve\)?)?)?\s*[:=-]?\s*([0-9,]+)\s*(?:total|beats?)\b"],
    )
    return ventricular, supraventricular


def first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".,;")
    return ""


def dedupe_findings(findings: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for finding in findings:
        key = finding.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(finding)
    return deduped


def extract_patient_profile(text: str) -> dict[str, str]:
    compact = re.sub(r"\s+", " ", text).strip()
    profile_match = re.search(
        r"Report for\s+(.+?)\s+ID\s+([A-Za-z0-9-]+)\s+Age\s+(\d{1,3}\s*(?:yrs?|years?)?)\s+Gender\s+([A-Za-z-]+)",
        compact,
        flags=re.IGNORECASE,
    )
    if profile_match:
        return {
            "name": profile_match.group(1).strip(),
            "id": profile_match.group(2).strip(),
            "age": profile_match.group(3).strip(),
            "gender": profile_match.group(4).strip(),
        }

    def find(pattern: str) -> str:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    profile = {
        "name": find(r"(?:patient name|name)\s*[:=-]\s*([A-Za-z ,.'-]+)"),
        "id": find(r"(?:patient id|mrn|medical record(?: number)?|id)\s*[:#=-]?\s*([A-Za-z0-9-]+)"),
        "age": find(r"age\s*[:=-]?\s*(\d{1,3}\s*(?:yrs?|years?)?)"),
        "gender": find(r"(?:gender|sex)\s*[:=-]?\s*(male|female|man|woman|nonbinary|non-binary|other)"),
    }
    return {key: value for key, value in profile.items() if value}


def extract_clinical_findings(evidence_texts: list[str]) -> list[str]:
    findings: list[str] = []
    for text in evidence_texts:
        for line in re.split(r"[\n\r]+", text):
            compact = re.sub(r"\s+", " ", line).strip(" -:")
            if len(compact) < 12:
                continue
            lowered = compact.lower()
            if any(
                term in lowered
                for term in [
                    "sinus rhythm",
                    "sinus tachycardia",
                    "sinus bradycardia",
                    "ectopic",
                    "avg hr",
                    "patient triggered",
                    "manual",
                    "summary report",
                ]
            ):
                findings.append(compact[:220])
    return dedupe_findings(findings)


def extract_unique_matches(text: str, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    deduped = []
    seen = set()
    for match in matches:
        compact = re.sub(r"\s+", " ", match).strip(" -:")
        key = compact.lower()
        if compact and key not in seen:
            seen.add(key)
            deduped.append(compact)
    return deduped
