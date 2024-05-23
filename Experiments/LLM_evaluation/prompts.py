PROMPT_FOR_LLAMA = False

PROMPTS = {
    # English zero-shot prompt
    "zero_shot_english": lambda context, question: create_prompt(
        language="english",
        context=context,
        question=question,
    ),
    # English Prompt & 5 examples from SQuAD v2 (english wiki)
    "five_shot_english": lambda context, question: create_prompt(
        language="english",
        context=context,
        question=question,
        examples_list=SQUAD_V2_EXAMPLES,
        examples_number_list=[0, 1, 2, 3, 4],
    ),
    # zero-shot prompt in German, Turkish, and Chinese
    "zero_shot_same_language": {
        "german": lambda context, question: create_prompt(
            language="german",
            context=context,
            question=question,
        ),
        "turkish": lambda context, question: create_prompt(
            language="turkish",
            context=context,
            question=question,
        ),
        "chinese": lambda context, question: create_prompt(
            language="chinese",
            context=context,
            question=question,
        ),
    },
    # German Prompt & 5 examples from german M2QA
    # To evaluate Chinese or Turkish data (cross-lingual)
    "five_shot_cross_lingual": {
        "german": {
            # German gets prompt in Chinese
            "creative_writing": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "news": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "product_reviews": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
        "turkish": {
            # Turkish gets prompt in German
            "creative_writing": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "news": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "product_reviews": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
        "chinese": {
            # Chinese gets prompt in Turkish
            "creative_writing": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "news": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            "product_reviews": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
    },
    # German Prompt & 5 German examples from M2QA in a different domain
    # To evaluate German data (cross-domain)
    "five_shot_cross_domain": {
        "german": {
            # Prompt for Creative Writing contains examples from german M2QA News
            "creative_writing": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for News contains examples from german M2QA Product Reviews
            "news": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for Product Reviews contains examples from german M2QA Creative Writing
            "product_reviews": lambda context, question: create_prompt(
                language="german",
                context=context,
                question=question,
                examples_list=M2QA_GERMAN_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
        "turkish": {
            # Prompt for Creative Writing contains examples from german M2QA News
            "creative_writing": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for News contains examples from german M2QA Product Reviews
            "news": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for Product Reviews contains examples from german M2QA Creative Writing
            "product_reviews": lambda context, question: create_prompt(
                language="turkish",
                context=context,
                question=question,
                examples_list=M2QA_TURKISH_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
        "chinese": {
            # Prompt for Creative Writing contains examples from german M2QA News
            "creative_writing": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_NEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for News contains examples from german M2QA Product Reviews
            "news": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_PRODUCT_REVIEWS_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
            # Prompt for Product Reviews contains examples from german M2QA Creative Writing
            "product_reviews": lambda context, question: create_prompt(
                language="chinese",
                context=context,
                question=question,
                examples_list=M2QA_CHINESE_CREATIVE_WRITING_EXAMPLES,
                examples_number_list=[0, 1, 2, 3, 4],
            ),
        },
    },
}

# Prompt derived from https://aclanthology.org/2023.findings-emnlp.878.pdf


######################################################
# # Llama prompt: chain of user - assistant messages instead dof having everything in the first prompt.
def create_prompt_llama(language, context, question, examples_list=None, examples_number_list=None):
    """Create a prompt with a specified number of examples."""

    if language == "english":
        system_prompt = ENGLISH_SYSTEM_PROMPT
        language_func = context_question_string_english
    elif language == "german":
        system_prompt = GERMAN_SYSTEM_PROMPT
        language_func = context_question_string_german

    # System prompt
    prompt = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # If there are examples, add them to the user prompt
    if examples_list is not None and examples_number_list is not None:
        for index in examples_number_list:
            example = examples_list[index]

            prompt.append(
                {
                    "role": "user",
                    "content": language_func(
                        example["context"],
                        example["question"],
                    ),
                }
            )

            prompt.append(
                {
                    "role": "assistant",
                    "content": example["answers"],
                },
            )

    prompt.append(
        {
            "role": "user",
            "content": language_func(context, question),
        }
    )

    return prompt


def create_prompt(language, context, question, examples_list=None, examples_number_list=None):
    """Create a prompt with a specified number of examples."""

    if PROMPT_FOR_LLAMA:
        return create_prompt_llama(language, context, question, examples_list, examples_number_list)

    if language == "english":
        system_prompt = ENGLISH_SYSTEM_PROMPT
        language_func = context_question_string_english
        answer_string = "\nAnswer: "
    elif language == "german":
        system_prompt = GERMAN_SYSTEM_PROMPT
        language_func = context_question_string_german
        answer_string = "\nAntwort: "
    elif language == "turkish":
        system_prompt = TURKISH_SYSTEM_PROMPT
        language_func = context_question_string_turkish
        answer_string = "\nCevap: "
    elif language == "chinese":
        system_prompt = CHINESE_SYSTEM_PROMPT
        language_func = context_question_string_chinese
        answer_string = "\n答案："

    # System prompt
    prompt = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    user_prompt = ""

    # If there are examples, add them to the user prompt
    if examples_list is not None and examples_number_list is not None:
        for index in examples_number_list:
            example = examples_list[index]
            user_prompt += language_func(
                example["context"],
                example["question"],
            )
            user_prompt += answer_string
            user_prompt += example["answers"]
            user_prompt += "\n\n"

    # Add the main user prompt
    user_prompt += language_func(context, question)
    user_prompt += answer_string

    prompt.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return prompt


def context_question_string_english(context, question):
    return f"Passage: {context}\nQuestion: {question}\nNote: Your answer should be directly extracted from the passage and be a single entity, name, or number, not a sentence. If the passage doesn't contain a suitable answer, respond with 'unanswerable'."


def context_question_string_german(context, question):
    return f"Textpassage: {context}\nFrage: {question}\nHinweis: Die Antwort sollte direkt aus dem Text extrahiert werden und aus einer einzelnen Entität, einem Namen oder einer Zahl bestehen, nicht aus einem Satz. Wenn die Textpassage keine passende Antwort enthält, antworte mit 'unbeantwortbar'."


def context_question_string_turkish(context, question):
    return f"Pasaj: {context}\nSoru: {question}\nNot: Cevabınız doğrudan pasajdan alınmalı ve cevabınız bir cümle değil, tek bir varlık, isim veya sayı olmalıdır. Eğer pasaj uygun bir cevap içermiyorsa, 'cevaplanamaz' şeklinde yanıt verin."


def context_question_string_chinese(context, question):
    return f'段落：{context}\n问题：{question}\n注意：您的答案应直接摘自文段，并且是一个单独的实体、名称或数字，而不是一个句子。如果文段中没有合适的答案，请回答 "无法回答"。'


ENGLISH_SYSTEM_PROMPT = "Task Description: Answer the question from the given passage. Your answer should be directly extracted from the passage, and it should be a single entity, name, or number, not a sentence. If the passage doesn't contain a suitable answer, please respond with 'unanswerable'."
GERMAN_SYSTEM_PROMPT = "Aufgabenbeschreibung: Beantworte die Frage anhand der gegebenen Textpassage. Die Antwort sollte direkt aus der Textpassage extrahiert werden und aus einer einzelnen Entität, einem Namen oder einer Zahl bestehen, nicht aus einem Satz. Wenn die Textpassage keine passende Antwort enthält, antworte bitte mit 'unbeantwortbar'."
TURKISH_SYSTEM_PROMPT = "Görev Tanımı: Soruyu verilen pasajdan yanıtlayın. Cevabınız doğrudan pasajdan alınmalı ve cevabınız bir cümle değil, tek bir varlık, isim veya sayı olmalıdır. Eğer pasaj uygun bir cevap içermiyorsa, lütfen 'cevaplanamaz' şeklinde yanıtlayın."
CHINESE_SYSTEM_PROMPT = '任务描述： 基于给定段落回答问题。答案应直接摘自文段，且应是一个实体、名称或数字，而不是一个句子。如果段落中没有合适的答案，请回答 "无法回答"。'

######################################################
# Examples for SQUAD v2 and M2QA (randomly selected)
SQUAD_V2_EXAMPLES = [
    # answerable
    {
        "id": "570953a7efce8f15003a7dff",
        "context": 'In 2007, BSkyB and Virgin Media became involved in a dispute over the carriage of Sky channels on cable TV. The failure to renew the existing carriage agreements negotiated with NTL and Telewest resulted in Virgin Media removing the basic channels from the network on 1 March 2007. Virgin Media claimed that BSkyB had substantially increased the asking price for the channels, a claim which BSkyB denied, on the basis that their new deal offered "substantially more value" by including HD channels and Video On Demand content which was not previously carried by cable.',
        "question": "What channels were removed from the network in March of 2007?",
        "answers": "the basic channels",
    },
    {
        "id": "5729e500af94a219006aa6b9",
        "context": "Following the Cretaceous–Paleogene extinction event, the extinction of the dinosaurs and the wetter climate may have allowed the tropical rainforest to spread out across the continent. From 66–34 Mya, the rainforest extended as far south as 45°. Climate fluctuations during the last 34 million years have allowed savanna regions to expand into the tropics. During the Oligocene, for example, the rainforest spanned a relatively narrow band. It expanded again during the Middle Miocene, then retracted to a mostly inland formation at the last glacial maximum. However, the rainforest still managed to thrive during these glacial periods, allowing for the survival and evolution of a broad diversity of species.",
        "question": "Savannah areas expanded over the last how many years?",
        "answers": "34 million",
    },
    {
        "id": "5729fb003f37b31900478628",
        "context": "It is conjectured that a progressive decline in hormone levels with age is partially responsible for weakened immune responses in aging individuals. Conversely, some hormones are regulated by the immune system, notably thyroid hormone activity. The age-related decline in immune function is also related to decreasing vitamin D levels in the elderly. As people age, two things happen that negatively affect their vitamin D levels. First, they stay indoors more due to decreased activity levels. This means that they get less sun and therefore produce less cholecalciferol via UVB radiation. Second, as a person ages the skin becomes less adept at producing vitamin D.",
        "question": "As a person gets older, what does the skin produce less of?",
        "answers": "vitamin D",
    },
    # unanswerable
    {
        "id": "5ad3f4b1604f3c001a3ff952",
        "context": "In 1066, Duke William II of Normandy conquered England killing King Harold II at the Battle of Hastings. The invading Normans and their descendants replaced the Anglo-Saxons as the ruling class of England. The nobility of England were part of a single Normans culture and many had lands on both sides of the channel. Early Norman kings of England, as Dukes of Normandy, owed homage to the King of France for their land on the continent. They considered England to be their most important holding (it brought with it the title of King—an important status symbol).",
        "question": "What battle took place in the 10th century?",
        "answers": "unanswerable",
    },
    {
        "id": "5ad4cd245b96ef001a10a112",
        "context": "Dendritic cells (DC) are phagocytes in tissues that are in contact with the external environment; therefore, they are located mainly in the skin, nose, lungs, stomach, and intestines. They are named for their resemblance to neuronal dendrites, as both have many spine-like projections, but dendritic cells are in no way connected to the nervous system. Dendritic cells serve as a link between the bodily tissues and the innate and adaptive immune systems, as they present antigens to T cells, one of the key cell types of the adaptive immune system.",
        "question": "What is named for its resemblance to dendritic cells?",
        "answers": "unanswerable",
    },
]

M2QA_GERMAN_CREATIVE_WRITING_EXAMPLES = [
    # answerable
    {
        "id": "de_books_0_91_q1",
        "question": "Was berührte sich?",
        "context": "Sie wünschte in diesem Augenblick, daß er ihr Komplimente sage, ihr den Hof mache, ja, sie wollte, wenn er's nicht von selbst that, herbeiführen, was ihre Gedanken und Sinne beschäftigte. So war es denn durchaus nicht ohne Absicht, daß sie, als er ihr näher trat, den Kopf so zur Seite neigte, daß seine Wange ihr Haar streifte, und ihre Häupter sich sanft berührten. Sie zog das ihrige auch nicht zurück, und als er gar absichtlich oder unabsichtlich sich leise an sie drängte, ließ sie es geschehen und wich erst nach einer Weile, ihm einen sinnverwirrenden Blick zuwerfend, zurück.",
        "answers": "ihre Häupter",
    },
    {
        "id": "de_books_3_15_q1",
        "question": "Wer ist aufgeregt?",
        "context": "\"O Gott, sehen Sie ihn nur an, guter Berner, ist mir doch, als sollte ich zu ihm gehen und fragen: Was fehlt dir, daß du nicht fröhlich bist mit den Fröhlichen? Wie gern wollte ich alles tun, dir zu helfen.\"-- Der Mensch denkt's, Gott lenkt's!!! Auch der Hofrat wurde jetzt unruhig; denn mit einem Ruck hatte sich der bleiche Fremde aufgerafft und stand nun in seiner ganzen Größe, in gebietender und doch graziöser Haltung da; aber sein Auge heftete sich furchtbar starrend nach der Saaltüre. Berner wollte eben aufstehen und zu ihm hin--",
        "answers": "Hofrat",
    },
    {
        "id": "de_books_0_61_q1",
        "question": "Was gibt es zu Essen?",
        "context": "Hochgeehrter Herr von Brecken! Sie haben unserer Tochter die liebenswürdige Zusage gemacht, uns besuchen zu wollen. Darauf hin bin ich so frei, Sie zu fragen, ob Sie ohne das Zeremoniell einer Antrittsvisite, auf die wir gern verzichten, im engen Familienkreise bei uns eine Suppe essen möchten. Wir würden darüber außerordentlich erfreut sein und bitten, gütigst dem Überbringer zu sagen, ob wir Sie um drei Uhr erwarten dürfen. Ihr sehr ergebener Konrad von Treffen",
        "answers": "eine Suppe",
    },
    # unanswerable
    {
        "id": "de_books_3_17_q3",
        "question": "Was klingt tief und zitternt?",
        "context": 'Die Nacht war grimmigkalt, der Himmel jetzt ganz rein, nur einzelne dunkle Wölkchen tanzten im Wirbel um den Mond. Schweigend schritten die beiden durch die Nacht der Kirche zu. Wenige Schritte, so standen sie am Portal des Münsters. Der Küster schrak zusammen, als dort aus dem Schatten eines Pfeilers eine hohe, in einen dunklen Mantel gehüllte Gestalt hervortrat. Es war jener Fremde, der Idas Interesse in so hohem Grade erregt hatte. "Schließ auf, schließ auf," sprach Martiniz, "denn es ist hohe Zeit!" Indem er sprach, fing es an zu surren und zu klappern, dumpf rollte gerade über ihnen im Turme das Uhrwerk, und in tiefen, zitternden Klängen schallte die zwölfte Stunde in die Lüfte.',
        "answers": "unbeantwortbar",
    },
    {
        "id": "de_books_0_46_q4",
        "question": "Wie alt musste Grete sein bis ihre Mutter nicht mehr vom Testament profitieren konnte?",
        "context": "„Weißt Du etwas von den Geldverhältnissen drüben?“ „Ja, man sagt, Herr von Tressen habe das ihm von seiner Frau mitgebrachte Vermögen bis auf den letzten Pfennig verthan, und beide lebten schon seit Jahren von Gretes Einkünften. Bis Grete ein bestimmtes Alter erreicht hatte, soll die Mutter auch testamentarisch Nutznießerin gewesen sein, seitdem aber keine Ansprüche mehr haben.“ „Ganz recht. Gleiches deutete schon der Verwalter Hederich an. — Wie beurteilt man ihn denn?“ „Man nennt ihn in der Umgegend ‚Drum und dran‘, weil er diese Worte stets an passender und unpassender Stelle gebraucht. Er ist ein einfacher aber sehr braver und von aller Welt geachteter Mann. Mein Vater hielt große Stücke auf ihn.“",
        "answers": "unbeantwortbar",
    },
]

# German
M2QA_GERMAN_NEWS_EXAMPLES = [
    # answerable
    {
        "id": "de_news_72_2_q0",
        "question": "Weshalb wurde Carlin Q. Williams verhaftet?",
        "context": "Der 39-jährige Carlin Q. Williams behauptet, der Sohn des am 21. April verstorbenen Prince Rogers Nelson zu sein. Williams sitzt gerade wegen Autodiebstahls und illegalen Waffenbesitzes in einem Knast in Colorado ein und begehrt einen DNA-Test. Prince soll 1976 seiner Mutter beigewohnt haben. Auch eine bis vor seinem Tod der Familie nicht bekannte Halbschwester des Stars ist aufgetaucht. So ein Erbe lässt Anwälte frohlocken. Zudem liegt noch kein abschließender Obduktionsbericht vor, der die Todesursache des 57-jährig verstorbenen Stars erklären würde. \n",
        "answers": "Autodiebstahls und illegalen Waffenbesitzes",
    },
    {
        "id": "de_news_8_2_q1",
        "question": "Warum schied Bayer Leverkusen, trotz eines Unentschiedens, aus?",
        "context": "Auch Valencia hat sich verabschiedet. Der Klub von Coach Gary Neville schied trotz eines 2:1-Heimsiegs gegen Athletic Bilbao aus, weil die erste Partie 0:1 verlorengegangen war und damit die Auswärtstorregel schlagend wurde. Valencia ging zwar durch Santi Mina (13.) und Aderlan Santos (37.) 2:0 in Führung, das Gegentor durch Aritz Aduriz (75.) besiegelte aber das Ende der Aufstiegshoffnungen. Das Aus kam auch für Bayer Leverkusen nach einem 0:0 gegen Villarreal. Die Spanier hatten das Hinspiel 2:0 gewonnen. Überraschend schaffte Sparta Prag den Viertelfinal-Einzug. Die Tschechen setzten sich bei Lazio Rom 3:0 durch, die Partie in Prag hatte 1:1 geendet.",
        "answers": "Die Spanier hatten das Hinspiel 2:0 gewonnen",
    },
    {
        "id": "de_news_112_1_q0",
        "question": "Wer wird vom Programm betreut?",
        "context": 'Projekt "MutMacher" der Jugendanwaltschaft bringt junge Flüchtlinge mit Mentoren zusammen. Salzburg – Jugendliche Flüchtlinge willkommen heißen, ihnen Halt geben und einfach für sie da sein – das ist das Ziel des Projekts MutMacher von der Kinder- und Jugendanwaltschaft (Kija) Salzburg. Freiwillige können sich bei der Kija melden und werden dann Paten eines Jugendlichen, der sich für das Projekt beworben hat. In Salzburg leben derzeit 166 unbegleitete minderjährige Flüchtlinge. Sie würden Mentoren besonders dringend brauchen, weil sie ganz ohne Bezugspersonen in Österreich sind, sagt die Salzburger Jugendanwältin Andrea Holz-Darenstaedt. ',
        "answers": "unbegleitete minderjährige Flüchtlinge",
    },
    # unanswerable
    {
        "id": "de_news_15_1_q4",
        "question": "Wieviele Chancen gab es in der zweiten Hälfte?",
        "context": "Kurz vor der Pause setzte Altach-Coach Damir Canadi schon zum Torjubel an. Nach einem weiten Pass von Alexander Pöllhuber tauchte Patrick Seeger allein vor Venturo auf, schupfte den Ball aber nicht nur über den Tormann, sondern auch über die Latte (42.). Die zweite Hälfte begann mit einer Chance auf beiden Seiten. Zunächst verfehlte Abel Camara in der 47. Minute per Kopf aus kurzer Distanz das Tor, wenige Sekunden später köpfelte Ngwat-Mahop über den Querbalken. Danach übernahmen die Altacher immer mehr die Initiative, Belenenses ließ sich zurückfallen.",
        "answers": "unbeantwortbar",
    },
    {
        "id": "de_news_11_0_q4",
        "question": "Werden ohne zusätzliche Anreize keine Mittel gegen tropische Krankheiten entwickelt?",
        "context": "Viel zu oft richte Entwicklungshilfe mehr Schaden an, als sie nutze. Den reichen Ländern habe schließlich auch niemand Vorschriften darüber gemacht, wie sie sich zu entwickeln hätten. Konkret hieße das: Produkte aus Entwicklungsländern leichter ins Land, lassen, weniger Zölle erheben. Die Industrieländer könnten auch ihren Pharmafirmen Anreize geben, um wirksame Mittel gegen Armutskrankheiten wie Malaria und Tuberkulose zu entwickeln. Geld direkt in die armen Länder zu schicken sei der falsche Weg. Länder mit guter Politik könnten ihre Armut selbst bekämpfen, Ländern mit schlechter Politik helfe auch das Geld nicht.",
        "answers": "unbeantwortbar",
    },
]

M2QA_GERMAN_PRODUCT_REVIEWS_EXAMPLES = [
    # answerable
    {
        "id": "de_review_206_q0",
        "question": "Was erleichtert das Aufbringen?",
        "context": "Das Aufbringen wird doch sehr durch das beigelegte Spray erleichtert - sollte es zu wenig sein (wenn ihr so grobmotorisch, wie ich seid) - hilft auch etwas Wasser. Leider fliegen bei mir auch diverse Katzenhaare und Staub von Teppichen durch die Wohnung. Daher findet sich dann doch mal das eine, oder andre Haar auf der Klebeseite - bloß nicht runter kratzen... dann hält die Folie nicht mehr. Die Folie dann unter Wasser halten und sanft abwaschen. Alles in allem sind die Reflexionen jetzt deutlich reduziert - nur direkte Reflexion der Sonne verwandelt das Display noch in eine weiße Fläche. Wenn man die Folie in die richtige Position gebracht hat, gibt es auch keine Probleme mit größeren Hüllen, wie denen von Spigen. Top\n",
        "answers": "das beigelegte Spray",
    },
    {
        "id": "de_review_190_q0",
        "question": "Was ist  an der Leiter zu kritisieren?",
        "context": "Die Leiter ist sehr instabil und wackelig. Man hat das Gefühl beim Ein- und Ausstieg gleich damit umzufallen. Was fehlt, ist ein Trittbrett zwischen Aussen- und Innenleiter. Sicher würde ich sie deshalb nicht nennen. Gerade bei Kindern würde ich sie immer festhalten. Ob das mit dem hochklappen so klappt, kann ich noch nicht beurteilen, da ich es noch nicht ausprobiert habe. Die Qualität des Materials ist in Ordnung. Deswegen hat es 2 Sterne gegeben. Kaufen würde ich sie mir aber nicht mehr.\n",
        "answers": "Die Leiter ist sehr instabil und wackelig",
    },
    {
        "id": "de_review_44_q1",
        "question": "Was braucht es für ein angenehmes clicken der Tasten?",
        "context": "Die Verarbeitung ist gut. Nach ersten Installation und 5 Minuten spielen habe ich sie aber wieder eingepackt. Warum? Die vorderen Maustasten haben so gut wie kein Feedback. Es fühlt sich an als würden sich die Tasten nicht bewegen obwohl die Aktion ausgeführt wird. Das Zweite ist, dass die Tasten für den Daumen für meinen Geschmack zu hoch sitzen und mit einer natürlichen Bewegung nicht zu erreichen sind. Alles in allem für meinen persönlichen Geschmack sehr ungemütlich zu spielen. Ich habe mich nun für die G203 von Logitec entschieden.",
        "answers": "Feedback",
    },
    # unanswerable
    {
        "id": "de_review_390_q4",
        "question": "Welches Bauteil hat gefehlt?",
        "context": "Die Blumentreppe kam mit allem Zubehör (auch kleines Werkzeug zum zusammenbauen). Die einzelnen Teile machen insgesamt einen stabilen Eindruck. Ein großer Minuspunkt sind leider – wie hier schon oft erwähnt – die Gewinde die etwas schief sind und es schwer machen die Schrauben reinzudrehen. Einmal zusammengebaut macht die Blumentreppe einen stabilen, schönen Eindruck. Leider kippelt meine etwas, da wohl eine Stange verzogen ist, das werde ich an den Füßchen ausgleichen.\n",
        "answers": "unbeantwortbar",
    },
    {
        "id": "de_review_303_q4",
        "question": "Woher kommt das Produkt?",
        "context": "Hundebesitzer werden es kennen, Produkte für Hunde sind nicht billig. Dabei möchte man seinem besten Freund das Beste bieten. Nach zahlreichen Tests und zahlreichen Rücksendungen sind wir bei diesem Produkt geblieben, da es weder zu teuer ist, noch Mängel in jeglicher Hinsicht aufweist. Leichte Handhabung, Gurt lässt sich einfach anschnallen, der Karabiner leicht bedienen. Der Gurt selbst weist keine Mängel auf, so dass wir am Ende sehr zufrieden sind mit diesem Einkauf. Nur zu empfehlen!\n",
        "answers": "unbeantwortbar",
    },
]

# Turkish
M2QA_TURKISH_CREATIVE_WRITING_EXAMPLES = [
    # answerable
    {
        "id": "books_tr_21_169_q0",
        "question": "Yazar Saudade adlı mekanı nasıl buluyor?",
        "context": '"Böyle dikilecek miyiz?" dedim beden biraz kısa olan bedene karşı. "Hı?" derken dalmış gibiydi. Yüzüme karşı. Sırıttım. "Bana aşık olmanın sırası değil Jeongguk,donarak ölmek istemiyorum." derken kolundan tutup sürüklemeye başlamıştım. "Kusacağım sanırım." demişti sözlerime karşı. Bu çocuk çok yalancıydı."Kus kus." dedim imayla."Nereye gideceğiz?" derken arkamdan yürümeyi bırakıp yanımda yerini almıştı. "Saudade?" dedim sorarcasına. Güzel ve kaliteli bir yerdi. "Hayır. Oranın çalışanları çok geriyor beni." dediğinde kıkırdadım. ',
        "answers": "Güzel ve kaliteli",
    },
    {
        "id": "books_tr_16_140_q2",
        "question": "Kim dosyalari iki dakikada hepsina dagitti?",
        "context": '" Biz varken sana bir şey olmaz." " Bok olmaz. Sizin yüzünüzden olacak. Keremle Yakalandım bir kaç kere. Gözlerine battı zaten. " Sırıtarak bana döndü. " Nasıl yakalandınız?" " Salak salak konuşma hadi çık. " Zar zor dışarı itip kapımı kapattım. İşimden edecekler beni ya. Allah Allah. Geldim burada adam gibi konsantre olup işimi yaomaya çalışıyorum, ilk defa bir işimde bu kadar sevilmişim, ondan da atılacağım onlar yüzünden. Yerime geçip dosyaları baştan düzenlemeye başladım. Berkan iki dakikada hepsini dağıtmıştı.',
        "answers": "Berkan",
    },
    {
        "id": "books_tr_1_12_q1",
        "question": "Ana karakter çalınan telefonu kime satmak istemiştir?",
        "context": 'Gözlerini kocaman açıp yattığı yerden hızla kalktı."Vay anasını! Nereden çaldın lan bu parayı?" Ona küçümseyici bir bakış attım.Bakışımı gören Ömer ellerini kaldırdı."Tamam, tamam mükemmel sen yaparsın."  Sırıttım. Tabi ki de yapardım."Bu parayı dertli bir beybabadan aldım. Kendisi çok bonkörmüş. Parasını ve telefonunu alırken çıtı çıkmadı." Ömer bana \'yav he he\' bakışı atıp yürümeye başladı.Ellerimdekini cebime yerleştirip yanında yürümeye başladım. "Hadi Rıfkıya gidelim de telefonu satalım." Bana kararsız bir bakış attı.',
        "answers": "Rıfkıya",
    },
    # unanswerable
    {
        "id": "books_tr_19_38_q3",
        "question": "İlacın bulundu odada kaçıncı kattadır?",
        "context": '"Pekala, tamam. İkiside odam da."Lütfen kreminizi sürüp sonrada ilacınızı alıp gelir misiniz?""Tamam."Hiç istemesede elindeki çantayı yere koyarak yukarı çıkmak için merdivenlere yöneldi."Yardım edeyim hyung. Elin yetişmeyecektir."Hoseok söylediğiyle gözlerini açarak şaşkın bir şekilde bana bakıp baş parmağını kaldırdı.Onlar tekrar yukarı çıkarken bende inince ilacını içmesi için mutfağa su getirmeye gittim."Teşekkür ederiz Hayel, iyi ki müdahale ettin.""Sen olmasaydın kesinlikle ne ilacı içecekti ne de kremi sürecekti."',
        "answers": "cevaplanamaz",
    },
    {
        "id": "books_tr_19_97_q3",
        "question": "Meyve sevmediğini ne zaman söylemişti?",
        "context": 'Dediği şeyle kaşlarım tekrar çatılırken haftalardır içimi kemiren soru onun yanındayken unutsam da şu an tekrar hatırlatmıştım. Nasıl gün geçtikçe hakkımda daha fazla şey öğreniyordu. Meyve sevmediğimi nereden biliyordu mesela?"Yanına dikkatlice oturdum ve elimi omzuna atıp sana yaklaşınca başını omzuma yasladın. Ben de burnumu saçlarına dayayıp aşık olduğum kokunu uzun uzun içime çektim. Arada da öpmeyi unutmuyordum tabi."Söylediğinin arkasından hafifçe güldüğünde sargıyı tuttuğum elim titremeye başlamıştı bile.',
        "answers": "cevaplanamaz",
    },
]

M2QA_TURKISH_NEWS_EXAMPLES = [
    # answerable
    {
        "id": "tr_news_138_0_q2",
        "question": "Basrolde oynayan oyuncu nereli?",
        "context": "'Burma'da Gözyaşları' filmine konu olmuştuAung San Suu Kyi'nin verdiği demokrasi mücadelesi, 1995 yapımı \"Beyond Rangoon\" (Burma'da Gözyaşları) filmine de konu olmuştu. Başrolünde ABD'li oyuncu Patricia Arquette'in oynadığı filmde, Myanmar'a giden bir Amerikalı turistin gözünden, 1988'deki demokrasi gösterileri anlatılıyor. Filmin dünya genelinde yarattığı etkinin ardından Myanmar'daki askeri cunta Suu Kyi'yi serbest bırakmıştı. Suu Kyi, serbest bırakılmasının ardından filmin yapımcılarına teşekkür etmişti.",
        "answers": "ABD'li",
    },
    {
        "id": "tr_news_151_1_q1",
        "question": "İkizlerin antrenörünün adı nedir?",
        "context": "Aynı His Maçta İyi Olmuyor Seda ve Eda Eroğlu'nun boks antrenörü Orhan Özaktı ise ikizlerin acı ve sevinci birlikte yaşamalarının maç sırasında bazı olumsuzluklara neden olduğunu belirtiyor. Orhan Özaktı hem bu durumu, hem de bulduğu çözüm formülünü şu sözlerle ifade ediyor: \"Seda rakibiyle dövüşürken, Eda kenarda bağırıp taktik vermeye çalışıyor. Eda bir darbe alsa Seda acıdan kıvranıyor. Benim konsantrasyonum bozuluyor ve taktik veremiyorum. Bu nedenle maç sırasında ikizlerin birini dışarı çıkarıyorum.",
        "answers": "Orhan Özaktı",
    },
    {
        "id": "tr_news2_2_1_q2",
        "question": "Bu donusum plani kimindir?",
        "context": "Engin Yeşil'in dönüşüm reçetesi: 1- Devlet ve inşaat firmaları bir araya gelerek dönüşüm planını yapmalı. 2- Kentsel dönüşüm yapılması planlanan bölgelerde önce ulaşım yatırımları olmalı. 3- Emsal en az 10 olmalı ve imar planı uygulamasında yükseklik serbest bırakılmalı. TOKİ, 39 noktada satış ofisi açtı Toplu Konut İdaresi Başkanlığı (TOKİ) , Türkiye genelinde satışa sunduğu projelerin satış ve tanıtımı için 39 tanıtım ve satış ofisi açtı. TOKİ, bu yolla ürettiği konutların daha hızlı satış ve pazarlamasını gerçekleştirmeyi hedefliyor.",
        "answers": "Engin Yeşil'in",
    },
    # unanswerable
    {
        "id": "tr_news_170_2_q3",
        "question": "de boer hangi kulubun teknik direktorudur?",
        "context": "Galatasaray'ın kaptanı Arda Turan'ın Avrupa'da oynayıp oynamayacağına ilişkin bir soru üzerine De Boer, \"Tabii ki fırsatlar olabilir. Önce kendisinin burada çok iyi geliştirmesi gerektiğini düşünüyorum. Bu her oyuncu için geçerlidir. Tabii iyi bir kulüp, güzel bir teklif getirdiğinde değerlendirmek gerekir. Türkiye'yi futbol anlamında uyuyan bir dev olarak görüyorum. Çok büyük bir potansiyel var ve zamanla stabil bir futbol ülkesine dönüşebilir. Herkes futbolu çok seviyor, ciddi bir futbol tutkusu var. Altyapıya da gereken yatırımları yapmak lazım\" ifadelerini kullandı.",
        "answers": "cevaplanamaz",
    },
    {
        "id": "tr_news2_31_1_q3",
        "question": "Basbakan Erdogan nereye ozel demec vermis?",
        "context": "Başbakan , 'nin 'ye olan borçlarının tamamının, 2012'nin son dönemi itibariyle ödenmiş olacağını söyledi. 'a özel bir demeç veren Başbakan Erdoğan, ekonomik konularla Türkiye'nin AB ve ile ilişkileri dahil çeşitli konulardaki soruları yanıtladı. Reuters, \"Haberleri oluşturanlar\" başlığıyla da Başbakan Erdoğan hakkında bir değerlendirme yazısı yayımladı. Erdoğan demecinde, \"IMF'ye 8 yıl önce, borcumuz 23,5 milyar idi, şu anda ise 6 milyar dolar. Zannediyorum 2012'nin sonuna doğru borçlarımızın tamamını kapatmış olacağız\" dedi.",
        "answers": "cevaplanamaz",
    },
]

M2QA_TURKISH_PRODUCT_REVIEWS_EXAMPLES = [
    # answerable
    {
        "id": "tr_review_288_q0",
        "question": "Hangi marka bu kaliteli telefonları üretiyor?",
        "context": "belki türkiye'de çok bilinen bir marka değil fakat marka sizi yanıltmasın bu marka çok kaliteli bir marka olup en uygun fiyata üreten bir markadır.diğer markaların en üst model(amiral gemisi) telefonlarından en büyük farklarından birisi fiyatıdır.bu fiyatta türkiye şartlarında bu telefonun yarı özelliğine sahip orta seviye telefonlar alınabiliyor.bu xiaomi firması en üst seviye kalite ve performansta telefon üretip en uygun fiyata satan bir firmadır.pil,performans,kamera,kalite,uygun fiyat hepsi bu akıllı telefonda var.",
        "answers": "xiaomi",
    },
    {
        "id": "tr_review_70_q2",
        "question": "Teslimat ve paketleme konusunda ürün nasıldır?",
        "context": "ürün aslında çok basit, dekoratif olarak güzel duruyor, rengi de güzel, ancak çok sağlam değil, ayakkabılık olarak kullanacaksanız ağır ayakkabıları(bot gibi) çok yığmayın derim. birleştirme ve sabitlemede kullanılan çiviler çaktığınız çıtaya göre çok büyük verilmiş, çakarken ya çıta kırılıyor ya da çiviler çıtanın arkasından çıkıyor. bir rafa üç ayakkabı sığıyor. toplamda 9 ayakkabı alır. kargo yine hızlı geldi, paketleme iyiydi. ben ürünü yeni alan biri olarak çıtaların çok ince olması ve kırılması dolayısıyla hoşlanmadım. olumlu değilim, daha sağlam bir ürün tercih edin derim",
        "answers": "kargo yine hızlı geldi, paketleme iyiydi",
    },
    {
        "id": "tr_review_69_q2",
        "question": "Ürünün fiyatı benzer ürünlerle kıyaslandığından nasıl?",
        "context": "fiyat olarak daha ucuzunu görmedim. üstelik kargo ücretsiz olması çok güzel. özellikle uzn bar ile çalışacaklar tercih etmeli. tam ölçmedim ama 2.5kg lık plakaların çapı sanırım 20cm falandı. ürünün 5kg lık plakaları eksik gelmişti onlarında muhtemelen 30-40cm arasında bir çapı vardır. uzun bar ile çalışırken sorun olmaz ama kısa barlarda çalışmak büyük sıkıntı. birde bu plakalar kum ağırlıklı ve diğer döküm plakalarla karşılaştırıldığında çok kullanışsız ve büyük. kum ağırlıklı plakaları araştırıp test edip öyle alın derim.",
        "answers": "daha ucuzunu görmedim",
    },
    # unanswerable
    {
        "id": "tr_review_4_q4",
        "question": "Trasa boyun bolgesinden baslamak neden onemli?",
        "context": "yaklaşık 4.5 sene önce aldim ve ilk traştan itibaren suana kadar hic sorun yaşamadim özellikle boyun alti kısımlarında jiletle sorun yaşayan varsa alsin. traşa once boyun bölgesinden baslamak gerekiyor bu önemli.suana kadar 2 bicak kullandim.yeni bicagi daha bugün aldim yani ortalama 2 seneye yakin gidiyor tek bicak. 2 günde 1 traş olan biriyim ve 3 güç seçeneği var ben ilk baslarda en düşük seviyede kullandim 1 yıl sonra da daha iyi kessin diye 2.devirde kullandim.mutlaka solüsyon kullanin temizliğinde. makinenin rahatlığını kullaninca anlarsiniz. daha önce mac 3 kullaniyordum.",
        "answers": "cevaplanamaz",
    },
    {
        "id": "tr_review_174_q4",
        "question": "Neden şebeke üzerinden görüntülü görüşme yapılamıyor?",
        "context": "cihaz ilk açıldığında varsayılan olarak telefonu çince kurmak istiyor. ama rahatlıkla türkçeyi yükleyebiliyorsunuz. hatta ben türkçe f klavye kullandığım için sorun olur sanmıştım. hiç sorunsuz f klavyeyi yükledim. kutudan çıkan şarj adaptörü türkiye uyumlu değil ama ithatatçı firma kutunun içine bir dönüştürücü araptör bırakmış. beni asıl şaşırtan ve hayal kırıklığına uğratan şebeke üzerinden görüntülü görüşme yapamıyorsunuz. ama whatsapp, line, bip gibi uygulamalarla rahatlıkla görüntülü görüşülebiliyor. artık almak size kalmış.",
        "answers": "cevaplanamaz",
    },
]

# Chinese
M2QA_CHINESE_CREATIVE_WRITING_EXAMPLES = [
    # answerable
    {
        "id": "books_zh_15_9_q2",
        "question": "文星伊洗澡的时候金容仙在做什么？",
        "context": '"姐姐要快点不然我要自己回去了～"浴室有两间，不过因为其中一间莲蓬头故障，文星伊先进去用最快速度洗完澡，因为外面等着的金容仙从她关门的那一刻就开始喊着害怕，现在已经换金容仙进去洗，她在外面等，不过那人还是同样的喊着害怕，好笑的想着到底要怎样才能不怕，为了安抚只好加大音量开玩笑，还用手机放音乐让金容仙别那么紧张。',
        "answers": "她在外面等",
    },
    {
        "id": "books_zh_54_29_q0",
        "question": "外面有什么？",
        "context": '帝夜瞳冷然地盯着她，黄金瞳散发着凌厉的冷意，"条件，你的名字。" 什么鬼？ 千璃唇角一阵抽搐。 她狠狠地甩了甩手腕，却发现无法动弹。 "你不是我的对手。" 见此，帝夜瞳冷然一笑，冰冷的视线扫视着她脑袋旁侧的警铃，"哪怕你侥幸逃掉，只要我一声令下，外面的保镖也会立即将你打成筛子。"',
        "answers": "保镖",
    },
    {
        "id": "books_zh_54_41_q2",
        "question": "帝夜瞳被撩后怎么了？",
        "context": '这个女人把他撩了，竟然敢说不小心？！ 千璃见他发火，心底也恼火十分，水眸狠狠一瞪帝夜瞳。 "老大不小的模样，装什么大处男，不就是一个细软短，坚持得了几分钟啊？" 帝夜瞳愣了愣，看着面前狂妄的丫头，冰冷的嘴角猛地一抽。 "女人，不要妄下定论。" 冷冽的双眸一眯，帝夜瞳强有力的臂膀把她禁锢，温热的湿气喷洒在耳畔，强势至极，"行不行，得上了再说！"',
        "answers": "发火",
    },
    # unanswerable
    {
        "id": "books_zh_54_32_q4",
        "question": "千璃自己有兵器吗？",
        "context": "帝夜瞳忽然有种揭开她面具的冲动。 尤其他的鼻尖，充斥着她发丝的独特清香。 然而，他确实也这么做了。 千璃微愣，急忙伸手想要把他推开，却发现一个硬物突然抵在了腰际。 尼玛！ 这家伙暗藏兵器！？ 千璃几乎在瞬间萎了。 饶是她身手再怎么厉害，那也不可能敌过近在咫尺的枪！ 帝夜瞳的手越来越近，千璃的心跳越来越快。",
        "answers": "无法回答",
    },
    {
        "id": "books_zh_54_23_q3",
        "question": "千璃长得怎么样？",
        "context": '这样的人，极度厌恶他人的靠近。 更别说和一个陌生女子上床！ 他现在，应该巴不得把自己撵走才对。 只是，等了好半天，千璃也没有等到自己想要的回答。 "帝少，那你要和人家亲亲吗~？" 她只好翘起了自己的红唇，毫无形象地给他下了一剂猛药。 帝夜瞳眸色一黯，锐利如刀地盯着她。 那帝王般的目光像是在说：你怎么还不给老子滚。',
        "answers": "无法回答",
    },
]

M2QA_CHINESE_NEWS_EXAMPLES = [
    # answerable
    {
        "id": "zh_news_116_0_q0",
        "question": "四想要做班子的什么？",
        "context": "四 是 要 做 班子 的 带头人 , 带头 讲 党性 、 重 品行 、 做 表率 , 带头 搞好 “ 三严三实 ” 专题 教育 , 带头 抓 班子 带队伍 , 带头 依法 办事 , 带头 廉洁 自律 , 带头 接受 党 和 人民 监督 , 带头 清清白白 做人 、 干干净净 做事 、 堂堂正正 做官 , 真正 做到 率先垂范 、 以上 率 下 。",
        "answers": "带头人",
    },
    {
        "id": "zh_news_36_2_q1",
        "question": "杨峰在读几年级？",
        "context": "月 12 日 中午 , 武强 实验 中学 学生 杨峰 从 衡水 中学 走 出来 。 “ 太难 了 ! ” 杨峰 皱 着 眉头 , 第一 句话 对 妈妈 这样 说 。 这 是 衡水 中学 对 武强 实验 中学 初三 年级 模拟 考试 排名 靠 前 学生 组织 的 一次 选拔 考试 。 虽然 中考 还 未 开考 , 但 衡水 中学 已经 开始 行动 。",
        "answers": "初三",
    },
    {
        "id": "zh_news_294_0_q2",
        "question": "网友有什么建议？",
        "context": "“ 我 妈妈 信 ‘ 全能神 ’ 有 3年 了 , ” 小猫 在 新 加入 的 群 里 说 , 目前 虽 未 出事 , 但 妈妈 越陷越深 还 劝 不 回 让 人 担忧 。 小猫 的 发言 很快 被 淹没 在 不断 闪 出 的 信息 中 , 开腔 的 网友 “ 家家有本难念的经 ” , 且 大都 与 “",
        "answers": "家家有本难念的经",
    },
    # unanswerable
    {
        "id": "zh_news_118_1_q3",
        "question": "在青岛死了几人？",
        "context": "京华时报 记者 韩旭 雷军 孟凡 泽 央视 报道 进展 爆炸 事故 9 相关 人员 被 控制 昨晚 11点 44 分 , 青岛市 黄岛区 宣传部 官方 微博 “ 黄岛 发布 ” 称 , 据悉 昨晚 警方 已 控制 此次 爆炸 事故 中石化 相关 人员 7 人 、 青岛 经济技术开发区 相关 人员 2 人 。 微博 中 没有 明确 提及 被 控制 人员 的 姓名 。",
        "answers": "无法回答",
    },
    {
        "id": "zh_news_73_2_q4",
        "question": "阳江监狱的职务犯有什么病？",
        "context": "广东省 监狱 管理局 表示 , 由于 职务 犯 资源 多 , 人脉 广 , 监狱 经常 会 遇到 “ 打招呼 ” 、 “ 找 关系 ” 的 执法 风险 , 职务 犯 也 更容易 享受 特殊 处 遇 。 为 规避 这些 执法 风险 , 广东 监狱 开始 对 职务 罪犯 实行 集中 关押 , 包括 县 处级 以上 干部 和 县处级 以下 的 部门 一 把手 。 阳江监狱 监狱 长林 映 坤 介绍 , 该 监狱 共 集中 关押 了 100 多名 职务 犯 , 多数 为 40 岁 以上 , 50 岁 占 主体 , 最大 为 68 岁 , 上至 正厅级 , 他们 中 多数 患有 高血压 、 糖尿病 、 心脏病 等 老年 慢性病 。",
        "answers": "无法回答",
    },
]

M2QA_CHINESE_PRODUCT_REVIEWS_EXAMPLES = [
    # answerable
    {
        "id": "zh_review_34_q1",
        "question": "麦教授的课堂主要内容是？",
        "context": "大侦探探案的方法，也能用来思考人类历史上的重大问题？甚至是用作日常的思维方法？对于年轻人来说，切实可用的合乎逻辑的推理方式是怎么样的？从一个看似不起眼的现象，怎么追踪它背后的原因，还原背后的故事。 麦教授教授了清晰简明的一堂普及课；对于我来说，这本小册子就是一套简明的思想体操——可以随着麦教授的思路，重温那些最基本的思维方法，比如怎么观察、拆解复杂社会现象，怎么运用环环紧扣的正向或反向推理方式得到合理的结论，怎样追踪线索，如何搜集证据（推理要用到的材料）。",
        "answers": "大侦探探案的方法",
    },
    {
        "id": "zh_review_72_q1",
        "question": "总共买了几本书？",
        "context": "不得不说，书不错，性价比很高！除此之外没有任何值得称道的了。换货换了2次，也就是说，因你贵公司根本不负责任的包装，摔坏了3本如此精美的书，其中一本现在还在我手上。 难道如此精美的书，不值得你们仔细包装一下吗？ 随随便便扔在一个根本不合适的箱子里，除此之外没有任何保护，再经过这么一路运输（物流的暴力程度想必大家都了解），书皮都和内页分开了，700多页的大部头。。。我看你们收回去怎么处理，Z秒杀？呵呵了。",
        "answers": "3本",
    },
    {
        "id": "zh_review_104_q2",
        "question": "杯子的保温功能如何？",
        "context": "等了半个月收到杯子那一刻我很无语的，杯身上碰撞的都是小点点，还有明显的凹点。骂人的心都有了。看别人的杯子送到都是好好的，偏偏我的是这样，还有说明书，别人还3种语言的，我一个都没有。订单下了之后，还怕把杯子刮花，还特意在网上订了一个杯套，唉。说客服吧，一说退货就扯到运费上面来了，国际物流贵且慢等等，虽然说杯子不影响使用，可是影响我的心情啊！！！！！！保温是很好，中午12点倒的开水进去，晚上6点还是原温度，第二天中午还是温的。",
        "answers": "保温是很好",
    },
    # unanswerable
    {
        "id": "zh_review_39_q4",
        "question": "快递在几号寄出？",
        "context": "确实是个挺像手镯的手表，小巧、漂亮、秀气，时间是有3、6、9、12，确实看起来不是太清楚。表带太长了，取了两三节后就带着合适了，但是，有个缺点，就是取下来的时候并不是那么容易抠开的，也不知道是否防水。。。快递还不错，寄到中国成都，12号下单，21号就拿到了，之前还总是担心会在生日之前都拿不到了呢，还好，物流给力！半年过后来看到，我发现居然好像是降价了！对了，手表不防水，我下雨天戴，结果表盘起了雾。",
        "answers": "无法回答",
    },
    {
        "id": "zh_review_59_q3",
        "question": "大冰的年龄多大？",
        "context": "看完了大冰的第四部书，因为你说你会读，所以也是第四次留下读后感。谢谢你，大冰，谢谢你的文字，谢谢你的陪伴。第一次翻看你的书时候, 我从英国跑回来和爸爸妈妈去医院拿妈妈的诊断结果，是妈妈刚刚确诊癌症，此后中国英国医院家里，无数次反复，无数个失眠夜里，无数次虐心的等待中，您的书中每个故事，每个人在陪伴我，有笑,有哭，最多的是感动。我和大冰刚好同岁，都是接近不惑年龄，说看你的书得到人生感悟是太捧杀你了，都是一把年纪都是有故事的人。从你的书里，读到最多的是共鸣与感动。好看依然，感动依旧。大冰，请继续装X牛X的生活，期待你的下一本书！",
        "answers": "无法回答",
    },
]
