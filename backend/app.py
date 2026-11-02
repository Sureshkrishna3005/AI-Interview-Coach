import io
import math
import os
from datetime import datetime
from xml.sax.saxutils import escape

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

from ollama_service import (
    DIMENSIONS,
    OLLAMA_MODEL,
    OllamaError,
    connection_status,
    evaluate_answer,
    generate_question,
    generate_summary,
)


app = Flask(__name__)
CORS(app)


def _interview_context(data):
    role = data.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Choose a role name before starting the interview.")

    interview_type = data.get("interviewType", "General")
    if not isinstance(interview_type, str) or interview_type not in {"General", "Technical", "Behavioral", "HR"}:
        raise ValueError("Choose a valid interview style.")
    difficulty = data.get("difficulty", "Medium")
    if not isinstance(difficulty, str) or difficulty not in {"Easy", "Medium", "Hard"}:
        raise ValueError("Choose Easy, Medium, or Hard difficulty.")
    experience = data.get("experience", "Entry level")
    if not isinstance(experience, str):
        raise ValueError("Experience level must be text.")

    return {
        "role": role.strip()[:120],
        "experience": experience[:80],
        "interviewType": interview_type,
        "difficulty": difficulty,
    }


def _question_context(data):
    context = _interview_context(data)
    number = data.get("questionNumber", 0)
    previous = data.get("usedQuestions", [])
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ValueError("Question number must be a non-negative integer.")
    if not isinstance(previous, list) or any(not isinstance(item, str) for item in previous):
        raise ValueError("Previous questions must be a list of strings.")
    context["questionNumber"] = number
    context["previousQuestions"] = previous[-20:]
    return context


def _evaluation_context(data):
    context = _question_context(data)
    for key in ("question", "expectedConcepts", "answer"):
        if key not in data:
            raise ValueError(f"Missing required evaluation field: {key}.")
    context.update({
        "question": data["question"],
        "expectedConcepts": data["expectedConcepts"],
        "answer": data["answer"],
    })
    return context


def _report_payload(data):
    answers = data.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ValueError("Complete at least one evaluated question before generating a report.")
    if any(not isinstance(item, dict) for item in answers):
        raise ValueError("The report contains an invalid answer entry.")
    if len(answers) > 20:
        raise ValueError("A report can contain at most 20 answers.")
    for entry in answers:
        if not isinstance(entry.get("question"), str) or not entry["question"].strip():
            raise ValueError("An answer is missing its interview question.")
        if not isinstance(entry.get("answer"), str) or not entry["answer"].strip():
            raise ValueError("An answer is missing the candidate response.")
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("Every answer must have a completed evaluation before the report can be generated.")
        scores = evaluation.get("dimensionScores")
        if not isinstance(scores, dict) or any(
            isinstance(scores.get(name), bool)
            or not isinstance(scores.get(name), (int, float))
            or not math.isfinite(scores[name])
            or not 0 <= scores[name] <= 100
            for name in DIMENSIONS
        ):
            raise ValueError("An answer evaluation is missing its dimension scores.")
        expected_overall = round(sum(scores[name] for name in DIMENSIONS) / len(DIMENSIONS))
        if evaluation.get("overallScore") != expected_overall:
            raise ValueError("An answer's overall score must equal the mean of its dimension scores.")
        for key in ("correctConcepts", "missingConcepts", "incorrectConcepts", "actionableFeedback"):
            if not isinstance(evaluation.get(key), list) or any(
                not isinstance(item, str) for item in evaluation[key]
            ):
                raise ValueError(f"An answer evaluation contains an invalid '{key}' list.")
    return answers


@app.get("/api/health")
def health():
    status = connection_status()
    return jsonify({
        "ok": status["status"] == "ready",
        "aiConfigured": status["status"] == "ready",
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        **status,
    })


@app.post("/api/interview")
def interview():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON request body is required."}), 400
    action = data.get("action")
    if not isinstance(action, str) or action not in {"start", "next", "evaluate", "finish"}:
        return jsonify({"error": "Unsupported interview action."}), 400

    try:
        if action in {"start", "next"}:
            result = generate_question(_question_context(data))
            number = data.get("questionNumber", 0)
            result["message"] = f"Question {number + 1}: {result['question']}"
            return jsonify({"mode": "ollama", **result})
        if action == "evaluate":
            result = evaluate_answer(_evaluation_context(data))
            return jsonify({"mode": "ollama", "evaluation": result})

        context = _interview_context(data)
        answers = _report_payload(data)
        summary = generate_summary(answers)
        dimension_scores = {
            name: round(sum(item["evaluation"]["dimensionScores"][name] for item in answers) / len(answers))
            for name in DIMENSIONS
        }
        score = round(sum(dimension_scores.values()) / len(dimension_scores))
        missing_concepts = []
        for item in answers:
            evaluation = item["evaluation"]
            missing_concepts.extend(
                concept for concept in evaluation.get("missingConcepts", [])
                if isinstance(concept, str) and concept not in missing_concepts
            )
        return jsonify({
            "mode": "ollama",
            "role": context["role"],
            "score": score,
            "dimensionScores": dimension_scores,
            "headline": summary["headline"],
            "strengths": summary["strengths"],
            "weaknesses": summary["weaknesses"],
            "improvements": summary["weaknesses"],
            "missingConcepts": missing_concepts,
            "nextSteps": summary["recommendations"],
            "recommendations": summary["recommendations"],
            "answers": answers,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OllamaError as exc:
        app.logger.warning("Ollama request failed: %s", exc)
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        app.logger.exception("Interview request failed unexpectedly.")
        return jsonify({"error": "The interview coach encountered an unexpected error. Please retry."}), 500


def _pdf_paragraph(value, style):
    text = escape(str(value or "")).replace("\n", "<br/>")
    return Paragraph(text, style)


@app.post("/api/report")
def download_report():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON report request is required."}), 400
    try:
        answers = _report_payload(data)
        report = data.get("report")
        if not isinstance(report, dict):
            raise ValueError("The completed performance report is missing.")
        candidate = data.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            candidate = "Candidate"
        role = data.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("The report is missing the job role.")

        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title="AI Interview Coach Performance Report",
            author="AI Interview Coach",
        )
        base = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=21, leading=26,
            textColor=colors.HexColor("#30275f"), alignment=TA_CENTER, spaceAfter=8,
        )
        section_style = ParagraphStyle(
            "SectionHeading", parent=base["Heading2"], fontSize=13, leading=17,
            textColor=colors.HexColor("#5141a0"), spaceBefore=14, spaceAfter=7,
        )
        body_style = ParagraphStyle(
            "ReportBody", parent=base["BodyText"], fontSize=9, leading=13, spaceAfter=5,
        )
        small_style = ParagraphStyle(
            "ReportSmall", parent=body_style, fontSize=8, leading=11,
        )
        story = [
            _pdf_paragraph("AI Interview Coach", title_style),
            _pdf_paragraph("Interview Performance Report", base["Heading2"]),
            Spacer(1, 8),
        ]
        information = [
            ["Candidate", candidate.strip()],
            ["Job role", role.strip()],
            ["Interview date", str(data.get("interviewDate") or datetime.now().strftime("%Y-%m-%d"))],
            ["Experience", str(data.get("experience") or "Not specified")],
            ["Interview style", str(data.get("interviewType") or "Not specified")],
            ["Difficulty", str(data.get("difficulty") or "Not specified")],
            ["Overall performance", f"{report.get('score', 0)}/100"],
        ]
        info_table = Table(information, colWidths=[1.55 * inch, 5.0 * inch])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0edff")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#403477")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dedbed")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([info_table, Spacer(1, 10)])
        if report.get("headline"):
            story.append(_pdf_paragraph(report["headline"], body_style))
        story.append(_pdf_paragraph("Overall dimension scores", section_style))
        dimension_rows = [["Dimension", "Score"]]
        for key, label in DIMENSIONS.items():
            dimension_rows.append([label, f"{report.get('dimensionScores', {}).get(key, 0)}/100"])
        dimensions_table = Table(dimension_rows, colWidths=[5.7 * inch, 0.85 * inch])
        dimensions_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5141a0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dedbed")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6fb")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([dimensions_table, _pdf_paragraph("Question-by-question review", section_style)])

        for index, entry in enumerate(answers, 1):
            evaluation = entry["evaluation"]
            block = [
                _pdf_paragraph(f"Question {index}: {entry['question']}", base["Heading3"]),
                _pdf_paragraph(f"<b>Your answer:</b> {entry['answer']}", body_style),
                _pdf_paragraph(
                    f"<b>Assessment:</b> {evaluation.get('assessment', 'not specified')} "
                    f"&nbsp;&nbsp; <b>Score:</b> {evaluation.get('overallScore', 0)}/100",
                    body_style,
                ),
                _pdf_paragraph(f"<b>Evaluation:</b> {evaluation.get('explanation', '')}", body_style),
            ]
            score_rows = [["Dimension", "Score"]] + [
                [label, f"{evaluation.get('dimensionScores', {}).get(key, 0)}/100"]
                for key, label in DIMENSIONS.items()
            ]
            score_table = Table(score_rows, colWidths=[5.7 * inch, 0.85 * inch])
            score_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ece9fb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dedbed")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            block.extend([
                score_table,
                _pdf_paragraph(f"<b>Correct concepts:</b> {', '.join(evaluation.get('correctConcepts', [])) or 'None identified'}", small_style),
                _pdf_paragraph(f"<b>Missing concepts:</b> {', '.join(evaluation.get('missingConcepts', [])) or 'None identified'}", small_style),
                _pdf_paragraph(f"<b>Incorrect concepts:</b> {', '.join(evaluation.get('incorrectConcepts', [])) or 'None identified'}", small_style),
                _pdf_paragraph(f"<b>Actionable feedback:</b> {'; '.join(evaluation.get('actionableFeedback', []))}", small_style),
                _pdf_paragraph(f"<b>Suggested improved answer:</b> {evaluation.get('improvedAnswer', '')}", body_style),
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dedbed")),
                Spacer(1, 8),
            ])
            story.append(KeepTogether(block))

        for title, key in (
            ("Strengths", "strengths"),
            ("Weaknesses", "weaknesses"),
            ("Missing concepts", "missingConcepts"),
            ("Final recommendations", "recommendations"),
        ):
            story.append(_pdf_paragraph(title, section_style))
            items = report.get(key, [])
            if not isinstance(items, list):
                items = []
            if not items:
                story.append(_pdf_paragraph("No items recorded.", body_style))
            for item in items:
                story.append(_pdf_paragraph(f"• {item}", body_style))

        document.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="interview-performance-report.pdf",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Could not generate the interview PDF report.")
        return jsonify({"error": "The PDF report could not be generated. Please retry."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
