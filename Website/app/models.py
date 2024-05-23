from enum import Enum

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Status(Enum):
    unassigned = 1
    assigned = 2
    finished = 3


class Annotator(db.Model):
    __tablename__ = "annotators"

    id = db.Column(db.String(30), primary_key=True)
    has_completed_tutorial = db.Column(db.Boolean)
    tutorial_start_time = db.Column(db.DateTime)
    tutorial_end_time = db.Column(db.DateTime)
    quality_checked = db.Column(db.Boolean)
    good_quality = db.Column(db.Boolean)

    annotation_session_list = db.relationship(
        "AnnotationSession", back_populates="annotator"
    )  # only exists if annotator is not deleted
    deleted_annotation_session_list = db.relationship(
        "DeletedAnnotationSession", back_populates="annotator"
    )  # only exists if annotator is deleted
    deleted_question_list = db.relationship(
        "DeletedQuestion", back_populates="annotator"
    )  # only exists if annotator is deleted

    def __repr__(self):
        return f"Annotator(id={self.id}, has_completed_tutorial={self.has_completed_tutorial}, tutorial_start_time={self.tutorial_start_time}, tutorial_end_time={self.tutorial_end_time}, quality_checked={self.quality_checked}, good_quality={self.good_quality})"  # fmt: skip


class Paragraph(db.Model):
    __tablename__ = "paragraphs"

    id = db.Column(db.Text, primary_key=True)
    doc_id = db.Column(db.Integer)
    domain = db.Column(db.String(30))
    original_subdomain = db.Column(db.String(30))
    original_id = db.Column(db.Text)
    language = db.Column(db.String(3))
    data_source = db.Column(db.String(30))
    date = db.Column(db.String(30))
    title = db.Column(db.Text)
    author = db.Column(db.Text)
    text = db.Column(db.Text)
    level = db.Column(db.String(30))
    status = db.Column(db.Enum(Status))
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    text_marked_as_faulty = db.Column(db.Boolean)

    annotation_session_id = db.Column(
        db.Integer, db.ForeignKey("annotation_sessions.id")
    )
    annotation_sessions = db.relationship(
        "AnnotationSession", back_populates="paragraph_list"
    )

    def __repr__(self):
        return f"Paragraph(id={self.id}, doc_id={self.doc_id}, domain={self.domain}, original_subdomain={self.original_subdomain}, original_id={self.original_id} language={self.language}, data_source={self.data_source}, date={self.date}, text={self.text}, level={self.level}, status={self.status}, annotation_session_id={self.annotation_session_id}, start_time={self.start_time}, end_time={self.end_time})"  # fmt: skip


class Question(db.Model):
    __tablename__ = "questions"

    paragraph_id = db.Column(
        db.Integer, db.ForeignKey("paragraphs.id"), primary_key=True
    )
    question_id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer_start = db.Column(db.Integer)
    answer_end = db.Column(db.Integer)
    answerable = db.Column(db.Boolean)
    answer_text = db.Column(db.Text)

    paragraph = db.relationship("Paragraph", backref="questions")

    def __repr__(self):
        return f"Question(paragraph_id={self.paragraph_id}, question_id={self.question_id}, question={self.question}, answer_start={self.answer_start}, answer_end={self.answer_end}, answerable={self.answerable}, answer_text={self.answer_text})"  # fmt: skip


class DeletedQuestion(db.Model):
    __tablename__ = "deleted_questions"

    paragraph_id = db.Column(
        db.Integer, db.ForeignKey("paragraphs.id"), primary_key=True
    )
    annotator_id = db.Column(
        db.String(30), db.ForeignKey("annotators.id"), primary_key=True
    )
    question_id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer_start = db.Column(db.Integer)
    answer_end = db.Column(db.Integer)
    answerable = db.Column(db.Boolean)
    answer_text = db.Column(db.Text)

    annotator = db.relationship("Annotator", back_populates="deleted_question_list")

    def __repr__(self):
        return f"DeletedQuestion(paragraph_id={self.paragraph_id}, annotator_id={self.annotator_id}, question_id={self.question_id}, question={self.question}, answer_start={self.answer_start}, answer_end={self.answer_end}, answerable={self.answerable}, answer_text={self.answer_text})"  # fmt: skip


class AnnotationSession(db.Model):
    __tablename__ = "annotation_sessions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    annotator_id = db.Column(db.String(30), db.ForeignKey("annotators.id"))
    annotator = db.relationship("Annotator", backref="paragraphs")
    study_id = db.Column(db.String(30), db.ForeignKey("prolific_studies.study_id"))
    study = db.relationship("ProlificStudy", backref="paragraphs")

    paragraph_list = db.relationship("Paragraph", back_populates="annotation_sessions")
    annotator = db.relationship("Annotator", back_populates="annotation_session_list")

    finished = db.Column(db.Boolean())
    prolific_session_id = db.Column(db.String(30))

    def __repr__(self):
        return f"AnnotationSession(id={self.id}, annotator_id={self.annotator_id}, study_id={self.study_id}, finished={self.finished}, prolific_session_id={self.prolific_session_id})"  # fmt: skip


class DeletedAnnotationSession(db.Model):
    __tablename__ = "deleted_annotation_sessions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    annotator_id = db.Column(db.String(30), db.ForeignKey("annotators.id"))
    study_id = db.Column(db.String(30), db.ForeignKey("prolific_studies.study_id"))
    finished = db.Column(db.Boolean())

    annotator = db.relationship(
        "Annotator", back_populates="deleted_annotation_session_list"
    )

    def __repr__(self):
        return f"DeletedAnnotation_Session(id={self.id}, annotator_id={self.annotator_id}, study_id={self.study_id}, finished={self.finished})"  # fmt: skip


class ProlificStudy(db.Model):
    __tablename__ = "prolific_studies"
    study_id = db.Column(db.String(30), primary_key=True)
    key = db.Column(db.String(30))

    def __repr__(self):
        return f"ProlificStudy(study_id={self.study_id}, key={self.key})"
