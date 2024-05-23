import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from flask.cli import AppGroup

from .models import (
    AnnotationSession,
    DeletedAnnotationSession,
    DeletedQuestion,
    Paragraph,
    Question,
    Status,
)

statistics_cli = AppGroup("statistics")

plt.style.use("seaborn-notebook")

DOMAINS = ["news", "review", "books"]
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


@statistics_cli.command("table_annotator_question_quality")
def create_table_annotator_question_quality():
    """
    Create a table with the number of annotators, questions, questions checked and questions quality
    """

    annotation_sessions = AnnotationSession.query.all()
    deleted_annotation_sessions = DeletedAnnotationSession.query.all()

    all_good_annotators = set(
        [
            annotation_session.annotator
            for annotation_session in annotation_sessions
            if annotation_session.annotator.quality_checked and annotation_session.annotator.good_quality
        ]
    )
    all_bad_quality_annotators = set(
        [annotation_session.annotator for annotation_session in deleted_annotation_sessions]
    )

    print("    \\begin{tabular}{l|cc|ccc}")
    print("        \\toprule")
    print(
        "        \\multicolumn{1}{c|}{Language} & \\multicolumn{2}{c|}{Annotators} &"
        " \\multicolumn{3}{c}{Questions} \\\\"
    )
    print("        ~ & Kept & Rejected & Kept & Rejected & Checked\\\\")
    for language in LANGUAGES:
        good_annotators = []
        bad_annotators = []

        good_annotators = [
            annotator
            for annotator in all_good_annotators
            if annotator.annotation_session_list[0].paragraph_list[0].language == language
        ]

        bad_annotators = [
            annotator
            for annotator in all_bad_quality_annotators
            if Paragraph.query.filter(Paragraph.id == annotator.deleted_question_list[0].paragraph_id).first().language
            == language
        ]

        questions_kept = Question.query.filter(Question.paragraph.has(language=language)).count()
        questions_rejected = DeletedQuestion.query.join(Paragraph).filter(Paragraph.language == language).count()
        questions_checked = 10 * (len(good_annotators) + len(bad_annotators))

        if questions_kept + questions_rejected == 0:
            questions_checked_percentage = 0
        else:
            questions_checked_percentage = questions_checked / (questions_kept + questions_rejected) * 100

        print("        \\midrule")
        print(
            f"        {LANGUAGES[language]} & {len(good_annotators)} &"
            f" {len(bad_annotators)} & {questions_kept} & {questions_rejected} &"
            f" {questions_checked} ({questions_checked_percentage:.2f}\%) \\\\ "
        )

    print("        \\bottomrule")
    print("    \end{tabular}")


@statistics_cli.command("annotator_statistic")
@click.argument("study_id", nargs=1, required=False)
def create_annotator_statistics(study_id=None):
    """
    Create plots showing the time distribution of annotators for the tutorial and the paragraphs
    """

    if study_id is not None:
        annotation_sessions = AnnotationSession.query.filter_by(study_id=study_id).all()
        deleted_annotation_sessions = DeletedAnnotationSession.query.filter_by(study_id=study_id).all()
    else:
        annotation_sessions = AnnotationSession.query.all()
        deleted_annotation_sessions = DeletedAnnotationSession.query.all()

    annotators = set([annotation_session.annotator for annotation_session in annotation_sessions])
    bad_quality_annotators = set(
        [annotation_session.annotator_id for annotation_session in deleted_annotation_sessions]
    )

    tutorial_times = []
    for annotator in annotators:
        if annotator.tutorial_end_time is not None and annotator.tutorial_start_time is not None:
            time_delta = annotator.tutorial_end_time - annotator.tutorial_start_time
            if time_delta.seconds < 1200:
                tutorial_times.append(time_delta.seconds)

    paragraph_times = []
    paragraphs = [
        paragraph for annotation_session in annotation_sessions for paragraph in annotation_session.paragraph_list
    ]

    for paragraph in paragraphs:
        if paragraph.end_time is not None and paragraph.start_time is not None:
            time_delta = paragraph.end_time - paragraph.start_time
            if time_delta.seconds < 600:
                paragraph_times.append(time_delta.seconds)

    quality_checked_annotators = [annotator for annotator in annotators if annotator.quality_checked]
    if len(quality_checked_annotators) > 0:
        percentage = (
            len(bad_quality_annotators) / (len(quality_checked_annotators) + len(bad_quality_annotators)) * 100.0
        )
        print(f"{percentage}% of annotators have bad quality")

    # Plotting
    plt.subplots(2, 1, tight_layout=True)

    plt.subplot(2, 1, 1)
    plt.hist(tutorial_times, bins=20, label="Histogram")
    plt.axvline(
        x=np.median(tutorial_times),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(tutorial_times)}s",
    )
    plt.axvline(
        x=np.mean(tutorial_times),
        color="yellow",
        linestyle="--",
        label=f"Mean: {np.mean(tutorial_times):.1f}s",
    )
    plt.title("Time taken for the tutorial")
    plt.xlabel("Time in seconds")
    plt.ylabel("Number of annotators")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(paragraph_times, bins=30, label="Histogram")
    plt.axvline(
        x=np.median(paragraph_times),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(paragraph_times)}s",
    )
    plt.axvline(
        x=np.mean(paragraph_times),
        color="yellow",
        linestyle="--",
        label=f"Mean: {np.mean(paragraph_times):.1f}s",
    )
    plt.title("Time per paragraph taken")
    plt.xlabel("Time in seconds")
    plt.ylabel("Number of paragraphs")
    plt.legend()
    plt.show()

    print(
        "Median time tutorial + 11 * median time per"
        f" paragraph:{(np.median(tutorial_times) + 11 * np.median(paragraph_times) ) / 60} minutes"
    )
    print("Median time 12 * median time per" f" paragraph:{12 * np.median(paragraph_times) / 60} minutes")


@statistics_cli.command("table_questions_finished_overview")
def table_questions_finished_overview():
    """
    Provides an overview of the amount of questions finished per domain per language
    """

    df = pd.DataFrame(
        columns=[domain.capitalize() for domain in DOMAINS],
        index=[language.capitalize() for language in LANGUAGES.values()],
    )

    df_unfinished = pd.DataFrame(
        columns=[domain.capitalize() for domain in DOMAINS],
        index=[language.capitalize() for language in LANGUAGES.values()],
    )

    for language in LANGUAGES:
        for domain in DOMAINS:
            df.loc[LANGUAGES[language].capitalize(), domain.capitalize()] = Paragraph.query.filter_by(
                domain=domain, language=language, status=Status.finished
            ).count()

            all_finished_paragraphs = Paragraph.query.filter_by(
                domain=domain, language=language, status=Status.finished
            ).all()

            unfinished_count = 0

            for paragraph in all_finished_paragraphs:
                annotation_session = AnnotationSession.query.filter_by(id=paragraph.annotation_session_id).first()
                if annotation_session is not None and not annotation_session.finished:
                    unfinished_count += 1

            df_unfinished.loc[LANGUAGES[language].capitalize(), domain.capitalize()] = unfinished_count

    print(f"Paragraphs:\n{df}\n")
    # to keep track of the current progress during annotation
    print(f"from unfinished sessions:\n{df_unfinished}\n")

    # Questions
    df_questions = pd.DataFrame(
        columns=[domain.capitalize() for domain in DOMAINS],
        index=[language.capitalize() for language in LANGUAGES.values()],
    )

    for language in LANGUAGES:
        for domain in DOMAINS:
            # Question has only paragraph_id and no information about domain and language
            # Thus we must go through the paragraphs
            unfiltered_questions = (
                Question.query.join(Paragraph)
                .filter(
                    Paragraph.domain == domain,
                    Paragraph.language == language,
                )
                .all()
            )

            # Filter with max word limit
            count = 0

            # This filter has no effect, because we already filter when the questions get submitted
            for question in unfiltered_questions:
                # don't include questions where the answers are too long
                if question.answerable:
                    if question.paragraph.language == "zh":
                        if (
                            len(
                                question.answer_text.replace(" ", "")
                                .replace(",", "")
                                .replace("，", "")
                                .replace(",", "")
                                .replace("。", "")
                                .replace("、", "")
                            )
                            > MAX_WORDS_PER_LANGUAGE["zh"]
                        ):
                            continue

                    elif len(question.answer_text.split(" ")) > MAX_WORDS_PER_LANGUAGE[question.paragraph.language]:
                        continue

                count += 1

            df_questions.loc[LANGUAGES[language].capitalize(), domain.capitalize()] = count

    print("Questions:")
    print(df_questions)


@statistics_cli.command("table_question_statistics")
def create_table_question_statistics():
    statistics = {language: {} for language in LANGUAGES}

    for language in LANGUAGES:
        questions = Question.query.filter(Question.paragraph.has(language=language)).all()
        answer_text_words = [len(question.answer_text.split(" ")) for question in questions]
        statistics[language]["mean words in answer"] = np.mean(answer_text_words)
        statistics[language]["median words in answer"] = np.median(answer_text_words)
        statistics[language]["answer_text_words"] = answer_text_words

    # Plot histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(list(LANGUAGES)):
            language = list(LANGUAGES)[i]
            ax.hist(statistics[language]["answer_text_words"], bins=20)
            ax.set_title(LANGUAGES[language])
            ax.set_xlabel("Number of words")
            ax.set_ylabel("Number of questions")
            ax.axvline(
                x=statistics[language]["median words in answer"],
                color="red",
                linestyle="--",
                label=f"Median: {statistics[language]['median words in answer']}",
            )
            ax.legend()

    fig.tight_layout()
    plt.show()

    for language in LANGUAGES:
        print(f"{LANGUAGES[language]}:")
        print("Median words: ", statistics[language]["median words in answer"])
        print("Mean words: ", statistics[language]["mean words in answer"])
        print("")


@statistics_cli.command("xquad_statistics")
def xquad_statistics():
    percentile = 97

    for language in ["en", "de", "tr"]:
        xquad_dataset = load_dataset("xquad", f"xquad.{language}")
        xquad_answer_words = [
            len(question["answers"]["text"][0].split(" ")) for question in xquad_dataset["validation"]
        ]

        percentile_value = np.percentile(xquad_answer_words, percentile)
        print(f"{language}: {percentile} percentile {percentile_value}")

    # For chinese: split by character
    xquad_dataset = load_dataset("xquad", "xquad.zh")
    xquad_answer_words = [
        len(
            question["answers"]["text"][0]
            .replace(" ", "")
            .replace(",", "")
            .replace("，", "")
            .replace(",", "")
            .replace("。", "")
            .replace("、", "")
        )
        for question in xquad_dataset["validation"]
    ]
    percentile_value = np.percentile(xquad_answer_words, percentile)
    print(f"chinese: {percentile} percentile {percentile_value}")


@statistics_cli.command("compare_german_with_xquad")
def create_comparison_german_with_xquad():
    """
    Compares the German questions-answer pairs with the XQuAD questions regarding the amount of words in the question and answer
    """

    for language in ["de", "zh"]:  # TODO: add "zh" when we have more data
        # Load XQuAD questions
        xquad_dataset = load_dataset("xquad", "xquad.de")

        # Load German questions with domains: news, books or review
        german_questions = (
            Question.query.join(Question.paragraph)
            .filter(
                Paragraph.language == "de",
                Paragraph.domain.in_(["news", "books", "review"]),
            )
            .filter(Question.answerable)
            .all()
        )

        # Get amount of words in question and answer
        xquad_question_words = [len(question["question"].split(" ")) for question in xquad_dataset["validation"]]
        xquad_answer_words = [
            len(question["answers"]["text"][0].split(" ")) for question in xquad_dataset["validation"]
        ]

        german_question_words = [len(question.question.split(" ")) for question in german_questions]
        german_answer_words = [len(question.answer_text.split(" ")) for question in german_questions]

        print(
            "Percentage of German answers with more than 8 words:"
            f" {sum([1 for answer in german_answer_words if answer > 10]) / len(german_answer_words) * 100}"
        )

        # Plot histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        axes[0].hist(xquad_question_words, bins=15)
        axes[0].set_title("XQuAD questions")
        axes[0].set_xlabel("Number of words")
        axes[0].set_ylabel("Number of questions")
        axes[0].axvline(
            x=np.median(xquad_question_words),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(xquad_question_words)}",
        )
        axes[0].axvline(
            x=np.mean(xquad_question_words),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(xquad_question_words)}",
        )
        axes[0].legend()

        axes[1].hist(xquad_answer_words, bins=20)
        axes[1].set_title("XQuAD answers")
        axes[1].set_xlabel("Number of words")
        axes[1].set_ylabel("Number of questions")
        axes[1].axvline(
            x=np.median(xquad_answer_words),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(xquad_answer_words)}",
        )
        axes[1].axvline(
            x=np.mean(xquad_answer_words),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(xquad_answer_words)}",
        )
        axes[1].legend()

        axes[2].hist(german_question_words, bins=15)
        axes[2].set_title("M2QA German questions (only answerable)")
        axes[2].set_xlabel("Number of words")
        axes[2].set_ylabel("Number of questions")
        axes[2].axvline(
            x=np.median(german_question_words),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(german_question_words)}",
        )
        axes[2].axvline(
            x=np.mean(german_question_words),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(german_question_words)}",
        )
        axes[2].legend()

        axes[3].hist(german_answer_words, bins=20)
        axes[3].set_title("M2QA German answers (only answerable)")
        axes[3].set_xlabel("Number of words")
        axes[3].set_ylabel("Number of questions")
        axes[3].axvline(
            x=np.median(german_answer_words),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(german_answer_words)}",
        )
        axes[3].axvline(
            x=np.mean(german_answer_words),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(german_answer_words)}",
        )
        axes[3].legend()

        fig.tight_layout()
        plt.show()
