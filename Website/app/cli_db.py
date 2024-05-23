import json
import pathlib

import click
import matplotlib.pyplot as plt
from flask.cli import AppGroup

from .models import (
    AnnotationSession,
    Annotator,
    Paragraph,
    ProlificStudy,
    Question,
    DeletedQuestion,
    DeletedAnnotationSession,
    Status,
    db,
)

db_cli = AppGroup("db")

plt.style.use("seaborn-notebook")

# "review" and "books" are the original names which we changed later on to "product_reviews" and "creative_writing"
DOMAINS = ["news", "review", "books"]  # "scientific", "legal"
LANGUAGES = {
    "de": "german",
    "zh": "chinese",
    "tr": "turkish",
}

MAX_WORDS_PER_LANGUAGE = {
    "en": 10,
    "de": 10,
    "tr": 9,
    "zh": 22,  # character count without spaces and punctuation
}


@db_cli.command("import")
@click.argument("filename")
def import_data(filename):
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            paragraph = Paragraph(
                id=data["id"],
                doc_id=data["doc_id"],
                domain=data["domain"],
                original_id=data["original_id"],
                language=data["language"],
                data_source=data["data_source"],
                date=data["date"],
                text=data["text"],
                original_subdomain=data.get("original_subdomain", None),
                title=data.get("title", None),
                level=data.get("level", None),
            )
            paragraph.status = Status.unassigned

            if Paragraph.query.filter_by(id=paragraph.id).first() is None:
                db.session.add(paragraph)

    db.session.commit()
    print("Data import completed.")


@db_cli.command("add-study")
@click.argument("study_id", nargs=1)
@click.argument("prolific_key", nargs=1)
def add_study(study_id, prolific_key):
    study = ProlificStudy(
        study_id=study_id,
        key=prolific_key,
    )
    db.session.add(study)
    db.session.commit()

    print(f'Added study "{study_id}" with Prolific key "{prolific_key}" successfully.')


@db_cli.command("export")
@click.argument("language", nargs=1)
@click.argument("domain", nargs=1)
def export_data(language, domain):
    if language not in LANGUAGES or domain not in DOMAINS:
        print(f"Language '{language}' or domain '{domain}' not supported.")
        return

    questions = Question.query.all()
    squad_style_export_data = []
    paragraph_ids = set()

    for question in questions:
        # Don't include questions from unsupported languages or domains
        # In the future we might want to include legal and scientific as well, but as of now we don't have data for these domains
        if (question.paragraph.language != language) or (question.paragraph.domain != domain):
            continue

        # Skip if paragraph not in paragraph_ids and paragraph_ids is already at 300
        if question.paragraph_id not in paragraph_ids and len(paragraph_ids) >= 300:
            continue

        squad_style_export_data.append(
            {
                "id": question.paragraph_id + "_q" + str(question.question_id),
                "question": question.question,
                "context": question.paragraph.text,
                "answers": (
                    {
                        "text": [question.answer_text],
                        "answer_start": [question.answer_start],
                    }
                    if question.answerable
                    else {"text": [], "answer_start": []}
                ),
            }
        )

        paragraph_ids.add(question.paragraph_id)

    # Rename "books" to "creative_writing" and "review" to "product_reviews"
    if domain == "books":
        file_name = "creative_writing.json"
    elif domain == "review":
        file_name = "product_reviews.json"
    else:
        file_name = f"{domain}.json"

    filepath = pathlib.Path("../m2qa_dataset") / LANGUAGES[language] / file_name
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for entry in squad_style_export_data:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")

    print(f"Data export completed to {filepath}")


@db_cli.command("annotator_aborted_study")
@click.argument("study_id", nargs=1)
@click.argument("prolific_pid", nargs=1)
def delete_session(study_id, prolific_pid):
    annotation_session = AnnotationSession.query.filter(
        AnnotationSession.annotator_id == prolific_pid,
        AnnotationSession.study_id == study_id,
    ).first()

    if annotation_session is None:
        print(f"no annotation session for '{prolific_pid}' in study '{study_id}'")
        return

    paragraphs = Paragraph.query.filter_by(annotation_session_id=annotation_session.id).all()
    has_annotated_something = False
    for paragraph in paragraphs:
        if paragraph.status != Status.finished:
            paragraph.annotation_session_id = None
            paragraph.status = Status.unassigned
        else:
            has_annotated_something = True
    db.session.commit()

    if not has_annotated_something:
        db.session.delete(annotation_session)
        db.session.commit()
        print(f"removed annotation session for '{prolific_pid}' in study '{study_id}'")
    else:
        print(f"unassigned all not yet finished paragraphs for '{prolific_pid}' in study" f" '{study_id}'")


@db_cli.command("purge_all_not_finished_sessions")
def purge_all_not_finished_sessions():
    # Ask for confirmation
    print("This will delete all annotation sessions that are not finished.")
    print("Are you sure you want to continue?")
    print("Type 'yes' to continue or anything else to abort.")
    confirmation = input()
    if confirmation != "yes":
        print("Aborting.")
        return

    annotation_sessions = AnnotationSession.query.filter(
        AnnotationSession.finished == False,
    ).all()

    number_of_deleted_sessions = 0
    number_of_deleted_paragraphs = 0
    number_of_deleted_questions = 0

    for annotation_session in annotation_sessions:
        # 1. Unassign all paragraphs
        paragraphs = Paragraph.query.filter_by(annotation_session_id=annotation_session.id).all()

        print(f"Unassigning {len(paragraphs)} paragraphs for session" f" {annotation_session.id}")

        for paragraph in paragraphs:
            paragraph.annotation_session_id = None
            paragraph.status = Status.unassigned
            number_of_deleted_paragraphs += 1

            # 2. Delete all questions for this paragraph. We don't move them to the deleted_questions as the deleted_questions only contains questions that were deleted due to pure quality.
            questions = Question.query.filter_by(paragraph_id=paragraph.id).all()
            for question in questions:
                db.session.delete(question)
                number_of_deleted_questions += 1

        # 3. Delete annotation session
        db.session.delete(annotation_session)
        number_of_deleted_sessions += 1

    db.session.commit()
    print("All not finished annotation sessions have been deleted.")
    print(f"Deleted {number_of_deleted_sessions} annotation sessions.")
    print(f"Deleted {number_of_deleted_paragraphs} paragraphs.")
    print(f"Deleted {number_of_deleted_questions} questions.")


@db_cli.command("blacklist")
def create_blacklist():
    blacklist = [
        annotator.id for annotator in Annotator.query.filter_by(quality_checked=True, good_quality=False).all()
    ]

    filename = "blacklist.txt"
    with open(filename, "w") as f:
        f.write(", ".join(blacklist))

    print(f"Data export completed to {filename}")


##################
# This command is only necessary, because we accidentally added some text passages from training data and now need to annotate these language-domain combinations again.
# We clear the full database except for the annotators table.


@db_cli.command("CRITICAL_prepare_database_for_new_study")
def prepare_database_for_new_study():
    # only keep annotators table
    # delete data of all other tables (i.e. paragraphs, annotation_sessions, questions, deleted_questions, deleted_annotation_sessions)

    # Ask for confirmation
    print("This will delete all data except for the annotators table.")
    print("Are you sure you want to continue?")
    print("Type 'yes' to continue or anything else to abort.")
    confirmation = input()
    if confirmation != "yes":
        print("Aborting.")
        return

    # delete deleted_questions
    deleted_questions = DeletedQuestion.query.all()
    for deleted_question in deleted_questions:
        db.session.delete(deleted_question)

    # delete deleted_annotation_sessions
    deleted_annotation_sessions = DeletedAnnotationSession.query.all()
    for deleted_annotation_session in deleted_annotation_sessions:
        db.session.delete(deleted_annotation_session)

    # delete questions
    questions = Question.query.all()
    for question in questions:
        db.session.delete(question)

    # delete paragraphs
    paragraphs = Paragraph.query.all()
    for paragraph in paragraphs:
        db.session.delete(paragraph)

    # delete annotation_sessions
    annotation_sessions = AnnotationSession.query.all()
    for annotation_session in annotation_sessions:
        db.session.delete(annotation_session)

    db.session.commit()

    print("All data except for the annotators table has been deleted.")
