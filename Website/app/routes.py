import logging
import os
import random
from datetime import datetime

import deepl
from app import app
from flask import jsonify, render_template, request

from .models import (
    AnnotationSession,
    Annotator,
    DeletedAnnotationSession,
    DeletedQuestion,
    Paragraph,
    ProlificStudy,
    Question,
    Status,
    db,
)

from .limits import LIMITS_ADDITIONAL_TO_TUTORIAL, LIMITS_WITHOUT_TUTORIAL

logging.basicConfig(
    filename="record.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


NUMBER_OF_QUESTIONS = 5
NUMBER_OF_ANSWERABLE_QUESTIONS = 3


MAX_WORDS_PER_LANGUAGE = {
    "en": 10,
    "de": 10,
    "tr": 9,
    "zh": 22,  # character count without spaces and punctuation
}


@app.route("/")
def overview():
    prolific_pid = request.args.get("PROLIFIC_PID")
    study_id = request.args.get("STUDY_ID")
    session_id = request.args.get("SESSION_ID")
    language = request.args.get("language")
    language_mapping = {"de": "German", "zh": "Chinese", "tr": "Turkish"}

    if not prolific_pid or not study_id or not session_id or not language or language not in language_mapping:
        app.logger.warning(f"{prolific_pid}: Overview: URL is missing parameters")
        return render_template(
            "overview.html",
            prolific_pid=prolific_pid,
            study_id=study_id,
            error_text=("The URL is missing parameters. Please go back to Prolific and click on" " the link again."),
            annotator_has_completed_tutorial=False,
            paragraph_ids=[],
            already_finished_paragraphs=[],
            number_of_paragraphs=0,
            task_language="",
        )

    study_doesnt_exist = ProlificStudy.query.get({"study_id": study_id}) is None
    if study_doesnt_exist:
        app.logger.warning(f"{prolific_pid}: Overview: Wrong study ID")
        return render_template(
            "overview.html",
            prolific_pid=prolific_pid,
            study_id=study_id,
            error_text=("The study_id is wrong. Please go back to Prolific and click on the" " link again."),
            annotator_has_completed_tutorial=False,
            paragraph_ids=[],
            already_finished_paragraphs=[],
            number_of_paragraphs=0,
            task_language="",
        )

    annotator_exists = db.session.query(Annotator.id).filter_by(id=prolific_pid).first() is not None

    # Create annotator if he isn't already in the db
    if not annotator_exists:
        annotator = Annotator(id=prolific_pid, has_completed_tutorial=False, quality_checked=False)
        db.session.add(annotator)
        db.session.commit()

    annotator_has_completed_tutorial = (
        db.session.query(Annotator.has_completed_tutorial).filter(Annotator.id == prolific_pid).scalar()
    )

    already_started_task = AnnotationSession.query.filter(
        AnnotationSession.annotator_id == prolific_pid,
        AnnotationSession.study_id == study_id,
    ).first()
    already_finished_paragraphs = []

    if already_started_task:
        paragraphs = already_started_task.paragraph_list
        for paragraph in paragraphs:
            if paragraph.status == Status.finished:
                already_finished_paragraphs.append(paragraph.id)
    else:
        if annotator_has_completed_tutorial:
            limits = LIMITS_WITHOUT_TUTORIAL[language]
        else:
            limits = LIMITS_ADDITIONAL_TO_TUTORIAL[language]

        # 1. get paragraphs
        paragraphs = []
        for domain in limits:
            if limits[domain] != 0:
                paragraphs += (
                    db.session.execute(
                        db.select(Paragraph)
                        .where(
                            Paragraph.language == language,
                            Paragraph.status == Status.unassigned,
                            Paragraph.domain == domain,
                        )
                        .limit(limits[domain])
                    )
                    .scalars()
                    .all()
                )

        # 2. check if we got all paragraphs.
        # Hint: it is beneficial to have more paragraphs than needed, as some annotators might abort and then other might join before we removed them via `flask db annotator_aborted_study <StudyID> <AnnotatorID>`
        expected_number_of_paragraphs = sum(v for v in limits.values())
        if len(paragraphs) < expected_number_of_paragraphs:
            app.logger.warning(f"{prolific_pid}: Study had already enough participants.")
            return render_template(
                "overview.html",
                prolific_pid=prolific_pid,
                study_id=study_id,
                error_text=(
                    "This study already had enough participants, sorry. Please go"
                    " back to Prolific and click                         on the"
                    " link to another of our studies."
                ),
                annotator_has_completed_tutorial=False,
                paragraph_ids=[],
                already_finished_paragraphs=[],
                number_of_paragraphs=0,
                task_language="",
            )

        for paragraph in paragraphs:
            paragraph.status = Status.assigned

        db.session.commit()

        # create Annotation_Session
        annotation_session = AnnotationSession(
            annotator_id=prolific_pid,
            study_id=study_id,
            paragraph_list=paragraphs,
            finished=False,
            prolific_session_id=session_id,
        )

        db.session.add(annotation_session)
        db.session.commit()

    annotator_has_completed_tutorial = (
        db.session.query(Annotator.has_completed_tutorial).filter(Annotator.id == prolific_pid).scalar()
    )

    paragraph_ids = [paragraph.id for paragraph in paragraphs]

    app.logger.info(
        f"{prolific_pid}: Overview page with: study_id={study_id},"
        f" session_id={session_id}, language={language},            "
        f" already_started_task={(already_started_task is not None)}"
    )

    return render_template(
        "overview.html",
        prolific_pid=prolific_pid,
        study_id=study_id,
        error_text=None,
        annotator_has_completed_tutorial=annotator_has_completed_tutorial,
        paragraph_ids=paragraph_ids,
        already_finished_paragraphs=already_finished_paragraphs,
        number_of_paragraphs=len(paragraph_ids),
        task_language=language_mapping[language],
    )


@app.route("/tutorial/step1.html")
def tutorial_step1():
    prolific_pid = request.args.get("PROLIFIC_PID")

    if not prolific_pid:
        app.logger.warning(f"{prolific_pid}: Tutorial step 1: URL is missing parameters")
        return render_template(
            "tutorial/step1.html",
            error_text=(
                'The URL is missing parameters. Please go back to the "Task Overview"'
                " page and click on                 the link for this tutorial task"
                " again."
            ),
            prolific_pid=prolific_pid,
        )

    annotator = Annotator.query.filter(Annotator.id == prolific_pid).first()
    annotator.tutorial_start_time = datetime.now()
    db.session.commit()

    app.logger.info(f"{prolific_pid}: Begins tutorial step 1")

    return render_template(
        "tutorial/step1.html",
        error_text=None,
        prolific_pid=prolific_pid,
    )


@app.route("/tutorial/step2.html")
def tutorial_step2():
    prolific_pid = request.args.get("PROLIFIC_PID")

    if not prolific_pid:
        app.logger.warning(f"{prolific_pid}: Tutorial step 2: URL is missing parameters")
        return render_template(
            "tutorial/step2.html",
            error_text=(
                'The URL is missing parameters. Please go back to the "Task Overview"'
                " page and click on                 the link for this tutorial task"
                " again."
            ),
            prolific_pid=prolific_pid,
        )

    app.logger.info(f"{prolific_pid}: Begins tutorial step 2")

    return render_template(
        "tutorial/step2.html",
        error_text=None,
        prolific_pid=prolific_pid,
    )


@app.route("/tutorial/step3.html")
def tutorial_step3():
    prolific_pid = request.args.get("PROLIFIC_PID")

    app.logger.info(f"{prolific_pid}: Begins tutorial step 3")

    if not prolific_pid:
        app.logger.warning(f"{prolific_pid}: Tutorial step 3: URL is missing parameters")
        return render_template(
            "tutorial/step3.html",
            error_text=(
                'The URL is missing parameters. Please go back to the "Task Overview"'
                " page and click on                 the link for this tutorial task"
                " again."
            ),
            prolific_pid=prolific_pid,
        )

    return render_template(
        "tutorial/step3.html",
        error_text=None,
        prolific_pid=prolific_pid,
    )


@app.route("/tutorial/submit", methods=["PUT"])
def tutorial_task_submission():
    submitted_data = request.get_json()
    prolific_pid = submitted_data["prolific_pid"]

    # Save in db that the annotator has completed the tutorial
    annotator = Annotator.query.filter(Annotator.id == prolific_pid).first()
    annotator.tutorial_end_time = datetime.now()
    annotator.has_completed_tutorial = True
    db.session.commit()

    app.logger.info(f"{prolific_pid}: Tutorial completed")

    return jsonify({"success": True}), 200


@app.route("/annotate")
def annotate():
    prolific_pid = request.args.get("PROLIFIC_PID")
    paragraph_id = request.args.get("paragraph_id")
    taskNumber = request.args.get("taskNumber")

    if not prolific_pid or not paragraph_id or not taskNumber:
        app.logger.warning(f"{prolific_pid}: Annotate: URL is missing parameters")
        return render_template(
            "annotate.html",
            taskNumber=taskNumber,
            error_text=(
                'The URL is missing parameters. Please go back to the "Task Overview"'
                " page and click on the                 link for this task again."
            ),
            has_completed_tutorial=True,
            prolific_pid=prolific_pid,
            paragraph_id=paragraph_id,
            paragraph_text="",
            answer_requirements_text="",
        )

    # Check that the annotator has successfully completed the tutorial
    annotator = Annotator.query.filter(Annotator.id == prolific_pid).first()
    if annotator is None or not annotator.has_completed_tutorial:
        app.logger.warning(
            f"{prolific_pid}: User tries to annotate before completing tutorial." f" Paragraph: {paragraph_id}"
        )
        return render_template(
            "annotate.html",
            taskNumber=taskNumber,
            error_text=("Before you start the tasks, you must complete the tutorial task."),
            prolific_pid=prolific_pid,
            paragraph_id=paragraph_id,
            paragraph_text="",
            answer_requirements_text="",
        )

    # Check that the annotator is actually assigned to this paragraph
    paragraph = Paragraph.query.filter(Paragraph.id == paragraph_id).first()

    if paragraph is None or paragraph.annotation_sessions.annotator_id != prolific_pid:
        app.logger.warning(f"{prolific_pid}: User tries to annotate unassigned paragraph" f" {paragraph_id}")
        return render_template(
            "annotate.html",
            taskNumber=taskNumber,
            error_text=(
                'This task has not been assigned to you. Please go back to the "Task'
                ' Overview" page and work                 on your next assigned task.'
            ),
            prolific_pid=prolific_pid,
            paragraph_id=paragraph_id,
            paragraph_text="",
            answer_requirements_text="",
        )

    if paragraph.status == Status.finished:
        app.logger.warning(f"{prolific_pid}: User clicked already annotated paragraph {paragraph_id}")
        return render_template(
            "annotate.html",
            taskNumber=taskNumber,
            error_text=(
                'You already completed this task. Please go back to the "Task Overview"'
                " page and work on                 your next assigned task."
            ),
            prolific_pid=prolific_pid,
            paragraph_id=paragraph_id,
            paragraph_text="",
            answer_requirements_text="",
        )

    language = paragraph.language
    if language == "zh":
        answer_requirements_text = (
            "The answers should be as short as possible and must be shorter than"
            f" {MAX_WORDS_PER_LANGUAGE['zh'] +1 } characters."
        )
    else:
        answer_requirements_text = (
            "The answers should be as short as possible and must be shorter than"
            f" {MAX_WORDS_PER_LANGUAGE[language] + 1} words. "
        )

    paragraph.start_time = datetime.now()
    db.session.commit()

    app.logger.info(f"{prolific_pid}: Start paragraph: {paragraph_id}")

    return render_template(
        "annotate.html",
        taskNumber=taskNumber,
        error_text=None,
        prolific_pid=prolific_pid,
        paragraph_id=paragraph_id,
        paragraph_text=paragraph.text,
        answer_requirements_text=answer_requirements_text,
    )


@app.route("/annotate/submit", methods=["PUT"])
def add_submission():
    submitted_data = request.get_json()

    prolific_pid = submitted_data["prolific_pid"]
    paragraph_id = submitted_data["paragraph_id"]
    annotations = submitted_data["annotation"]

    qa_pairs = []

    all_identifier_names = [a["from_name"] for a in annotations]

    # get all qa-pairs
    for i in range(1, NUMBER_OF_QUESTIONS + 1):
        question_identifier = "question-" + str(i)
        answer_identifier = "answer-" + str(i)
        if question_identifier not in all_identifier_names:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Please write a text for question " + str(i),
                    }
                ),
                200,
            )

        if answer_identifier not in all_identifier_names:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Please select an answer span for question " + str(i),
                    }
                ),
                200,
            )

        question = [a for a in annotations if a["from_name"] == question_identifier][0]
        answer = [a for a in annotations if a["from_name"] == answer_identifier][0]

        qa_pairs.append(
            {
                "question": question["value"]["text"][0],
                "answer": {
                    "startOffset": answer["value"]["start"],
                    "endOffset": answer["value"]["end"],
                },
                "answerable": i <= NUMBER_OF_ANSWERABLE_QUESTIONS,
            }
        )

    # Check that each question has a different answer span
    for i, qa_pair in enumerate(qa_pairs):
        for j, qa_pair2 in enumerate(qa_pairs[i + 1 :]):
            if qa_pair["question"] == qa_pair2["question"]:
                if qa_pair["answer"] == qa_pair2["answer"]:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": (
                                    f"Question {i + 1} and {i + j + 2} have the same"
                                    " answer highlighted. Please delete the"
                                    " highlighted answer of both and highlight"
                                    " different answers. If you have any prolems,"
                                    " write us via Prolific."
                                ),
                            }
                        ),
                        200,
                    )

    # Verifying that the annotator is actually assigned to this task.
    # This way, no one can change paragraphs that they are not allowed to change.
    paragraph = Paragraph.query.get(paragraph_id)
    if paragraph.annotation_sessions.annotator_id != prolific_pid:
        return (
            jsonify(
                {
                    "success": False,
                    "message": (
                        "You are working on a task that wasn't assigned to you. Please"
                        ' go back to the                         "Task Overview" page'
                        " and work on your next assigned task."
                    ),
                }
            ),
            200,
        )

    # Check length of answerable questions
    for question_number, qa_pair in enumerate(qa_pairs):
        if qa_pair["answerable"]:
            answer_start = qa_pair["answer"]["startOffset"]
            answer_end = qa_pair["answer"]["endOffset"]
            answer_text = paragraph.text[answer_start:answer_end]
            if paragraph.language == "zh":
                characters_in_answer = len(
                    answer_text.replace(" ", "")
                    .replace(",", "")
                    .replace(",", "")
                    .replace("。", "")
                    .replace("、", "")
                    .replace("，", "")
                )
                if characters_in_answer > MAX_WORDS_PER_LANGUAGE["zh"]:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": (
                                    f"The answer of question {question_number + 1} is"
                                    " too long. Please make it shorter than"
                                    f" {MAX_WORDS_PER_LANGUAGE['zh'] +1 } characters."
                                    " Remember that the answers should be as short as"
                                    " possible. Currently the answer is"
                                    f" {characters_in_answer} characters long"
                                ),
                            }
                        ),
                        200,
                    )

            elif len(answer_text.split(" ")) > MAX_WORDS_PER_LANGUAGE[paragraph.language]:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": (
                                f"The answer of question {question_number + 1} is too"
                                " long. Please make it shorter than"
                                f" {MAX_WORDS_PER_LANGUAGE[paragraph.language] +1 } words."
                                " Remember that the answers should be as short as"
                                " possible. Currently the answer is"
                                f" {len(answer_text.split(' '))} words long"
                            ),
                        }
                    ),
                    200,
                )

    text_marked_as_faulty = "quality" in all_identifier_names

    # store annotated questions in db
    paragraph.status = Status.finished
    paragraph.end_time = datetime.now()
    paragraph.text_marked_as_faulty = text_marked_as_faulty

    for question_number, qa_pair in enumerate(qa_pairs):
        question: Question = Question.query.get({"paragraph_id": paragraph_id, "question_id": question_number})
        answer_start = qa_pair["answer"]["startOffset"]
        answer_end = qa_pair["answer"]["endOffset"]
        answer_text = paragraph.text[answer_start:answer_end]

        if question is not None:
            question.question = qa_pair["question"]
            question.answer_start = answer_start
            question.answer_end = answer_end
            question.answerable = qa_pair["answerable"]
            question.answer_text = answer_text
        else:
            question = Question(
                paragraph_id=paragraph_id,
                question_id=question_number,
                question=qa_pair["question"],
                answer_start=answer_start,
                answer_end=answer_end,
                answerable=qa_pair["answerable"],
                answer_text=answer_text,
            )
            db.session.add(question)

    db.session.commit()

    app.logger.info(f"{prolific_pid}: Completed paragraph {paragraph_id}")

    return jsonify({"success": True}), 200


@app.route("/finish", methods=["PUT"])
def finish():
    submitted_data = request.get_json()

    prolific_pid = submitted_data["prolific_pid"]
    study_id = submitted_data["study_id"]

    # 1. check if study exists
    study_doesnt_exist = ProlificStudy.query.get({"study_id": study_id}) is None
    if study_doesnt_exist:
        return (
            jsonify(
                {
                    "success": False,
                    "message": (
                        "An error has occurred. Please close this page and open it via"
                        " the link from Prolific                                 again."
                        " After that, try submitting again. If this error persists,"
                        " please write us via                                 Prolific."
                    ),
                }
            ),
            200,
        )

    # 2. Check if the annotator has annotated all assigned paragraphs
    annotation_session = AnnotationSession.query.filter(
        AnnotationSession.annotator_id == prolific_pid,
        AnnotationSession.study_id == study_id,
    ).first()
    for paragraph in annotation_session.paragraph_list:
        if paragraph.status != Status.finished:
            return jsonify({"success": False, "paragraph": paragraph.id}), 200

    # 3. return key
    prolific_key = ProlificStudy.query.get(study_id).key

    annotation_session.finished = True
    db.session.commit()

    app.logger.info(f"{prolific_pid}: Completed the study {study_id}. Receives key: {prolific_key}")

    return jsonify({"success": True, "prolific_key": prolific_key}), 200


@app.route("/quality.html", methods=["GET"])
def quality():
    return render_template("quality.html")


@app.route("/quality/get_next_annotator", methods=["GET"])
def quality_get_next_annotator():
    annotation_session = (
        AnnotationSession.query.filter(AnnotationSession.finished.is_(True))
        .join(Annotator)
        .filter(Annotator.quality_checked.is_not(True))
        .first()
    )

    number_of_annotators_left = (
        db.session.query(Annotator)
        .join(AnnotationSession, isouter=True)
        .filter(AnnotationSession.finished.is_(True), Annotator.quality_checked.is_not(True))
        .distinct(Annotator.id)
        .count()
    )

    if annotation_session is None:
        return (
            jsonify({"success": False, "message": "All annotators have been checked."}),
            200,
        )
    else:
        annotator = annotation_session.annotator
        annotation_sessions = AnnotationSession.query.filter(AnnotationSession.annotator_id == annotator.id).all()

        paragraph_ids = [
            paragraph.id
            for annotation_session in annotation_sessions
            for paragraph in annotation_session.paragraph_list
        ]

        questions = Question.query.filter(Question.paragraph_id.in_(paragraph_ids)).all()

        # 1. select a random answerable and unaswerable question
        randomly_selected_answerable_question = random.sample(
            [question for question in questions if question.answerable], 1
        )
        randomly_selected_unanswerable_question = random.sample(
            [question for question in questions if not question.answerable], 1
        )

        # 2. select 8 more random questions
        randomly_selected_questions = random.sample(
            [
                question
                for question in questions
                if question.question_id != randomly_selected_answerable_question[0].question_id
                and question.question_id != randomly_selected_unanswerable_question[0].question_id
            ],
            8,
        )
        number_of_annotation_sessions = len(annotation_sessions)

        # Create DeepL translator
        # File deepl_auth_key.txt is in the same directory
        file = os.path.join(os.path.dirname(__file__), "deepl_auth_key.txt")
        with open(file, "r") as f:
            auth_key = f.read()
        translator = deepl.Translator(auth_key)

        annotated_data = []

        for question in (
            randomly_selected_answerable_question
            + randomly_selected_unanswerable_question
            + randomly_selected_questions
        ):
            text = question.paragraph.text
            question_text = question.question
            answer = question.answer_text

            next_data = {
                "text": text,
                "question": question_text,
                "answer": answer,
                "answerable": question.answerable,
                "answer_start": question.answer_start,
                "answer_end": question.answer_end,
            }

            if question.paragraph.language == "zh" or question.paragraph.language == "tr":
                # DeepL does not support all languages
                source_language = question.paragraph.language.upper()
                next_data["translated"] = {
                    "text": translator.translate_text(text, source_lang=source_language, target_lang="EN-GB").text,
                    "question": translator.translate_text(
                        question_text, source_lang=source_language, target_lang="EN-GB"
                    ).text,
                    "answer": translator.translate_text(answer, source_lang=source_language, target_lang="EN-GB").text,
                }

            annotated_data.append(next_data)

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "annotator_id": annotator.id,
                        "number_of_annotation_sessions": number_of_annotation_sessions,
                        "annotated_data": annotated_data,
                        "number_of_annotators_left": number_of_annotators_left,
                    },
                }
            ),
            200,
        )


@app.route("/quality/checked_annotator", methods=["POST"])
def checked_annotator():
    submitted_data = request.get_json()
    annotator_id = submitted_data["annotator_id"]
    good_quality = submitted_data["good_quality"]

    annotator = Annotator.query.get(annotator_id)
    annotator.quality_checked = True
    annotator.good_quality = good_quality
    if not good_quality:
        annotation_sessions = AnnotationSession.query.filter(AnnotationSession.annotator_id == annotator_id).all()
        for annotation_session in annotation_sessions:
            paragraph_ids = [paragraph.id for paragraph in annotation_session.paragraph_list]

            # Move annotation session to the DeletedAnnotationSessions table
            deleted_annotation_session = DeletedAnnotationSession(
                annotator_id=annotator_id,
                study_id=annotation_session.study_id,
                finished=annotation_session.finished,
            )
            db.session.add(deleted_annotation_session)
            db.session.delete(annotation_session)

            questions = Question.query.filter(Question.paragraph_id.in_(paragraph_ids)).all()

            for question in questions:
                deleted_question = DeletedQuestion(
                    paragraph_id=question.paragraph_id,
                    annotator_id=annotator_id,
                    question_id=question.question_id,
                    question=question.question,
                    answer_start=question.answer_start,
                    answer_end=question.answer_end,
                    answerable=question.answerable,
                    answer_text=question.answer_text,
                )
                db.session.add(deleted_question)
                db.session.delete(question)

            for paragraph in Paragraph.query.filter(Paragraph.id.in_(paragraph_ids)).all():
                paragraph.status = Status.unassigned

    db.session.commit()

    return jsonify({"success": True}), 200
