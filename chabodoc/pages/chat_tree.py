from numpy import *
import random


def get_response_tree_1(tree_id, case, tag):
    # tree 1
    if case == 0:
        if tag == "good words":
            response = "Das freut mich zu hören! Hast du aktuell viel zu erledigen?"
            case = 1
        else:
            response = "Oh, das tut mir leid. Was beschäftigt dich denn gerade?"
            case = 2

    elif case == 1:
        if tag == "bad words":
            response = "Das ist auch mal ganz angenehm, nicht so viel tun zu müssen. Wie läuft dein Tag bisher?"
        else:
            response = "Ich hoffe, dass es noch nicht zu viel ist und dass du genug Kraft hast, um alle deine Aufgaben zu erledigen. Wie läuft dein Tag bisher?"
        case = 1
        tree_id = 2

    elif case == 2:
        if tag == "bad words":
            response = "Das ist schön, dass dich nicht so viel beschäftigt. Wie läuft dein Tag bisher?"
            case = 1
            tree_id = 2
        else:
            response = "Aktuell ist ja auch viel los. Willst du mir noch mehr darüber erzählen?"
            case = 2.1

    elif case == 2.1:
        if tag == "bad words":
            response = "Das ist okay. Wie läuft dein Tag bisher?"
            case = 1
            tree_id = 2
        else:
            response = "Okay, dann erzähl mal."
            case = 2.2

    elif case == 2.2:
        response = "Ja, das kann ich verstehen. Ich hoffe, dass dich bald weniger belastet! Wie läuft dein Tag bisher?"
        case = 1
        tree_id = 2

    return response, tree_id, case


def get_response_tree_2(tree_id, case, tag, prediction):
    # tree 2
    if case == 1:
        if tag == "bad words":
            response = "Oh, was ist denn heute blöd gewesen? :/"
            case = 2
        else:
            if tag == "good words":
                response = (
                    prediction
                    + " Ich habe mal ein paar Fragen zu deinem Studium. Hast du eigentlich Kurse in Präsenz?"
                )
            else:
                response = "Ah okay, verstehe. Ich habe mal ein paar Fragen zu deinem Studium. Hast du eigentlich Kurse in Präsenz?"
            case = 1
            tree_id = 3

    elif case == 2:
        response = "Oh, ja, das kann ich verstehen. Ich habe mal ein paar Fragen zu deinem Studium. Hast du eigentlich Kurse in Präsenz?"
        case = 1
        tree_id = 3

    return response, tree_id, case


def get_response_tree_3(tree_id, case, tag, prediction):
    # tree 3
    if case == 1:
        if tag == "bad words":
            response = "Achso, also nur online Kurse. Wie war denn das Online-Learning für dich letztes Semester?"
            case = 6
        else:
            response = "Dann hast du bestimmt trotzdem Online-Uni gehabt. Wie war denn das Online-Learning für dich letztes Semester?"
            case = 2

    elif case == 2:
        response = "Und wie geht es dir aktuell mit den Hybrid-Unterrichtsformaten?"
        case = 3

    elif case == 3:
        if tag == "bad words":
            response = "Welche Nachteile siehst du denn?"
        else:
            response = "Welche Vorteile siehst du denn?"
        case = 4

    elif case == 4:
        response = (
            "Machst du dir Sorgen bezüglich Covid-19, wenn du vor Ort sein musst?"
        )
        case = 5

    elif case == 5:
        if tag == "bad words":
            response = "Da bin ich ja beruhigt. Während der Pandemie ist das Studentenleben ja stark verändert. Fühlst du dich in deinem sozialen Leben innerhalb des Studiums eingeschränkt?"
        else:
            response = "Ich hoffe, du machst dir nicht zu große Sorgen! Während der Pandemie ist das Studentenleben ja stark verändert. Fühlst du dich in deinem sozialen Leben innerhalb des Studiums eingeschränkt?"
        case = 1
        tree_id = 4

    elif case == 6:
        response = (
            prediction
            + " Während der Pandemie ist das Studentenleben ja stark verändert. Fühlst du dich in deinem sozialen Leben innerhalb des Studiums eingeschränkt?"
        )
        case = 1
        tree_id = 4

    return response, tree_id, case


def get_response_tree_4(tree_id, case, tag):
    # tree 4
    if case == 1:
        if tag == "bad words":
            response = "Das freut mich zu hören :) Lernst du trotz Covid-19 aktuell in einer Lerngruppe?"
            case = 3
        else:
            response = "Das tut mir leid :( Fühlst du dich oft alleine?"
            case = 2

    elif case == 2:
        if tag == "bad words":
            response = "Das ist schade :( Die Pandemie ist echt hart..."
        else:
            response = "Ah, okay, verstehe. Die Pandemie ist echt hart... Lernst du trotz Covid-19 aktuell in einer Lerngruppe?"
        case = 3

    elif case == 3:
        response = "Ah okay. Wie war denn der Kontakt zu deinen Dozierenden? Hattest du letztes Semester Kontakt zu deinen Dozierenden?"
        case = 1
        tree_id = 5

    return response, tree_id, case


def get_response_tree_5(tree_id, case, tag):
    # tree 5
    if case == 1:
        if tag == "good words":
            response = "War der Kontakt ausreichend?"
            case = 2
        else:
            response = "Oh, hättest du dir mehr Kontakt gewünscht?"
            case = 3

    elif case == 2 or case == 3:
        if case == 2 and tag == "good words":
            response = "Das freut mich zu hören! Wie war denn die allgemeine Organisation von deinem Studium: Warst du letztes Semester gut informiert, was du tun musstest?"
        elif case == 3 and tag == "good words":
            response = "Okay, danke für die Antwort! Wie war denn die allgemeine Organisation von deinem Studium: Warst du letztes Semester gut informiert, was du tun musstest?"
        else:
            response = "Okay, das leite ich mal weiter. Wie war denn die allgemeine Organisation von deinem Studium: Warst du letztes Semester gut informiert, was du tun musstest?"
        case = 0
        tree_id = 6

    return response, tree_id, case


def get_response_tree_6(tree_id, case, tag):
    # tree 6
    if case == 0:
        if tag == "good words":
            response = "Okay. Dann kanntest du auch alle deine Termine, zum Beispiel von Prüfungen?"
            case = 1
        else:
            response = "Oh, das ist natürlich ärgerlich :/ In diesem Semester war ja auch die Lehre ganz anders. Glaubst du denn, dass du auf den Alltag als Arzt gut vorbereitet wirst?"
            case = 1
            tree_id = 7

    elif case == 1:
        if tag == "bad words":
            response = "Mhh, okay. In diesem Semester war ja auch die Lehre ganz anders. Glaubst du denn, dass du auf den Alltag als Arzt gut vorbereitet wirst?"
        else:
            response = "Das ist gut! In diesem Semester war ja auch die Lehre ganz anders. Glaubst du denn, dass du auf den Alltag als Arzt gut vorbereitet wirst?"
        case == 1
        tree_id = 7

    return response, tree_id, case


def get_response_tree_7(tree_id, case, tag, prediction):
    # tree 7
    if case == 1:
        response = (
            prediction
            + " Eine Frage hätte ich noch: Hältst du Schauspielpatienten für eine gute Alternative zu echten Patienten?"
        )
        case = 0
        tree_id = 8

    return response, tree_id, case


def get_response_tree_8(tree_id, case, tag, prediction):
    # tree 8
    if case == 0:
        if tag == "good words":
            response = "Freut mich, dass das für dich gut klappt :) Hast du das Gefühl, dass du weißt, was dich inhaltlich dieses Semester erwartet?"
        else:
            response = "Verstehe. Hast du das Gefühl, dass du weißt, was dich inhaltlich dieses Semester erwartet?"
        case = 1

    elif case == 1:
        response = (
            prediction
            + " Und weißt du, was dich dieses Semester ablauftechnisch erwartet?"
        )
        case = 2

    elif case == 2:
        response = (
            "Alles klar. Würdest du allgemein sagen, dass dich dein Studium stresst?"
        )
        case = 0
        tree_id = 9

    return response, tree_id, case


def get_response_tree_9(tree_id, case, tag):
    # tree 9
    if case == 0:
        if tag == "good words":
            response = "Ohje, das klingt ja nicht so gut :( Was stresst dich denn?"
            case = 1
        else:
            response = "Das ist schön zu hören. Und was machst du denn in deiner Freizeit (zum Beispiel ein Hobby)?"
            case = 2
            tree_id = 10

    elif case == 1:
        response = "Das klingt sehr stressig... Aber hast du ein Hobby, das du in deiner Freizeit machst?"
        case = 4
        tree_id = 10

    return response, tree_id, case


def get_response_tree_10(tree_id, case, tag):
    # tree 10
    if case == 2:
        response = "Wie hilft dir dein Hobby beim entspannen?"
        case = 3

    elif case == 3:
        if tag == "bad words":
            response = "Das ist ja schade, dass du damit nicht entspannen kannst. Aber danke, jetzt hast du auch erstmal alle meine Fragen beantwortet! Hast du sonst noch was auf dem Herzen?"
        else:
            response = "Das klingt nach einem schönen Hobby. Ich hänge viel zu viel in Chatrooms ab. Und danke, jetzt hast du auch erstmal alle meine Fragen beantwortet! Hast du sonst noch etwas, das du mir erzählen möchtest?"
        case = 1
        tree_id = 11

    elif case == 4:
        if tag == "bad words":
            response = "Mhh, das ist ja schade. Aber danke, jetzt hast du auch erstmal alle meine Fragen beantwortet! Hast du sonst noch was auf dem Herzen?"
            case = 1
            tree_id = 11
        else:
            response = "Was ist denn dein Hobby?"
            case = 5

    elif case == 5:
        response = "Das klingt nach einem schönen Hobby. Ich verbringe meine Freizeit am liebsten mit chatten! Und danke, jetzt hast du auch erstmal alle meine Fragen beantwortet! Willst du noch was loswerden?"
        case = 1
        tree_id = 11

    return response, tree_id, case


def get_response_tree_11(tree_id, case, tag, finished_chat):
    # Tree 11
    if case == 1:
        if tag == "bad words":
            response = "Alles klar :) Vielen Dank für das Beantworten meiner Fragen und dass du es so lange mit mir ausgehalten hast! Machs gut!"
            finished_chat = True
            case = 3
        else:
            response = "Dann erzähl doch mal :)"
            case = 2
    elif case == 2:
        ans = random.choice(
            [
                "Aha. Interessant!",
                "Spannend.",
                "Ach so.",
                "Hmm...",
                "Sowas habe ich ja noch nie gehört!",
                "Sachen gibts.",
            ]
        )
        ques = random.choice(
            [
                " Hast du sonst noch etwas, das du mir erzählen möchtest?",
                " Gibt es noch mehr zu erzählen?,"
                " Hast du sonst noch was auf dem Herzen?",
                " Willst du noch was loswerden?",
            ]
        )
        response = ans + ques
        case = 1

    return response, tree_id, case, finished_chat


def answer_tree(tree_id, case, tag, prediction, finished_chat):
    if tree_id == 1:
        response, tree_id, case = get_response_tree_1(tree_id, case, tag)
    elif tree_id == 2:
        response, tree_id, case = get_response_tree_2(tree_id, case, tag, prediction)
    elif tree_id == 3:
        response, tree_id, case = get_response_tree_3(tree_id, case, tag, prediction)
    elif tree_id == 4:
        response, tree_id, case = get_response_tree_4(tree_id, case, tag)
    elif tree_id == 5:
        response, tree_id, case = get_response_tree_5(tree_id, case, tag)
    elif tree_id == 6:
        response, tree_id, case = get_response_tree_6(tree_id, case, tag)
    elif tree_id == 7:
        response, tree_id, case = get_response_tree_7(tree_id, case, tag, prediction)
    elif tree_id == 8:
        response, tree_id, case = get_response_tree_8(tree_id, case, tag, prediction)
    elif tree_id == 9:
        response, tree_id, case = get_response_tree_9(tree_id, case, tag)
    elif tree_id == 10:
        response, tree_id, case = get_response_tree_10(tree_id, case, tag)
    elif tree_id == 11:
        response, tree_id, case, finished_chat = get_response_tree_11(
            tree_id, case, tag, finished_chat
        )

    return response, tree_id, case, finished_chat
