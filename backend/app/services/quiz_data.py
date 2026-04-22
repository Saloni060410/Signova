"""
Static question bank.
Each level has 5 sub-quizzes × 5 questions = 25 questions per level.
"""

from typing import Dict, List

QUIZ_BANK: Dict[str, Dict[int, List[dict]]] = {
    # ── BEGINNER ──────────────────────────────────────────────────────────────
    "beginner": {
        1: [
            {"id": 1, "type": "static", "sign": "A", "options": ["A", "B", "C", "D"], "correct": 0},
            {"id": 2, "type": "static", "sign": "B", "options": ["A", "B", "C", "D"], "correct": 1},
            {"id": 3, "type": "static", "sign": "C", "options": ["A", "B", "C", "D"], "correct": 2},
            {"id": 4, "type": "static", "sign": "D", "options": ["A", "B", "C", "D"], "correct": 3},
            {"id": 5, "type": "static", "sign": "E", "options": ["D", "E", "F", "G"], "correct": 1},
        ],
        2: [
            {"id": 6,  "type": "static", "sign": "F", "options": ["E", "F", "G", "H"], "correct": 1},
            {"id": 7,  "type": "static", "sign": "G", "options": ["E", "F", "G", "H"], "correct": 2},
            {"id": 8,  "type": "static", "sign": "H", "options": ["F", "G", "H", "I"], "correct": 2},
            {"id": 9,  "type": "static", "sign": "I", "options": ["G", "H", "I", "J"], "correct": 2},
            {"id": 10, "type": "static", "sign": "J", "options": ["H", "I", "J", "K"], "correct": 2},
        ],
        3: [
            {"id": 11, "type": "static", "sign": "K", "options": ["J", "K", "L", "M"], "correct": 1},
            {"id": 12, "type": "static", "sign": "L", "options": ["K", "L", "M", "N"], "correct": 1},
            {"id": 13, "type": "static", "sign": "M", "options": ["L", "M", "N", "O"], "correct": 1},
            {"id": 14, "type": "static", "sign": "N", "options": ["M", "N", "O", "P"], "correct": 1},
            {"id": 15, "type": "static", "sign": "O", "options": ["N", "O", "P", "Q"], "correct": 1},
        ],
        4: [
            {"id": 16, "type": "static", "sign": "P", "options": ["O", "P", "Q", "R"], "correct": 1},
            {"id": 17, "type": "static", "sign": "Q", "options": ["P", "Q", "R", "S"], "correct": 1},
            {"id": 18, "type": "static", "sign": "R", "options": ["Q", "R", "S", "T"], "correct": 1},
            {"id": 19, "type": "static", "sign": "S", "options": ["R", "S", "T", "U"], "correct": 1},
            {"id": 20, "type": "static", "sign": "T", "options": ["S", "T", "U", "V"], "correct": 1},
        ],
        5: [
            {"id": 21, "type": "static", "sign": "U", "options": ["T", "U", "V", "W"], "correct": 1},
            {"id": 22, "type": "static", "sign": "V", "options": ["U", "V", "W", "X"], "correct": 1},
            {"id": 23, "type": "static", "sign": "W", "options": ["V", "W", "X", "Y"], "correct": 1},
            {"id": 24, "type": "static", "sign": "X", "options": ["W", "X", "Y", "Z"], "correct": 1},
            {"id": 25, "type": "static", "sign": "Z", "options": ["X", "Y", "Z", "A"], "correct": 2},
        ],
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    "medium": {
        1: [
            {"id": 101, "type": "static", "sign": "Hello",    "options": ["Hello", "Goodbye", "Thank You", "Please"], "correct": 0},
            {"id": 102, "type": "static", "sign": "Thank You","options": ["Sorry", "Please", "Thank You", "Hello"],   "correct": 2},
            {"id": 103, "type": "static", "sign": "Sorry",    "options": ["Help", "Sorry", "Stop", "Go"],             "correct": 1},
            {"id": 104, "type": "video",  "sign": "Please",   "options": ["Please", "Thanks", "Yes", "No"],           "correct": 0},
            {"id": 105, "type": "static", "sign": "Goodbye",  "options": ["Hello", "Stop", "Goodbye", "Help"],        "correct": 2},
        ],
        2: [
            {"id": 106, "type": "static", "sign": "Yes",  "options": ["Yes", "No", "Maybe", "Stop"],          "correct": 0},
            {"id": 107, "type": "video",  "sign": "No",   "options": ["Yes", "No", "Help", "Please"],         "correct": 1},
            {"id": 108, "type": "static", "sign": "Help", "options": ["Stop", "Go", "Help", "Wait"],          "correct": 2},
            {"id": 109, "type": "static", "sign": "Good", "options": ["Bad", "Good", "Great", "Okay"],        "correct": 1},
            {"id": 110, "type": "video",  "sign": "Bad",  "options": ["Good", "Bad", "Neutral", "Terrible"],  "correct": 1},
        ],
        3: [
            {"id": 111, "type": "static", "sign": "Stop",  "options": ["Go", "Stop", "Wait", "Run"],   "correct": 1},
            {"id": 112, "type": "video",  "sign": "Go",    "options": ["Go", "Stop", "Wait", "Run"],   "correct": 0},
            {"id": 113, "type": "static", "sign": "Wait",  "options": ["Go", "Stop", "Wait", "Run"],   "correct": 2},
            {"id": 114, "type": "static", "sign": "Love",  "options": ["Hate", "Love", "Like", "Miss"],"correct": 1},
            {"id": 115, "type": "video",  "sign": "Like",  "options": ["Hate", "Love", "Like", "Miss"],"correct": 2},
        ],
        4: [
            {"id": 116, "type": "static", "sign": "Eat",    "options": ["Eat", "Drink", "Sleep", "Walk"], "correct": 0},
            {"id": 117, "type": "video",  "sign": "Drink",  "options": ["Eat", "Drink", "Sleep", "Walk"], "correct": 1},
            {"id": 118, "type": "static", "sign": "Sleep",  "options": ["Eat", "Drink", "Sleep", "Walk"], "correct": 2},
            {"id": 119, "type": "static", "sign": "Walk",   "options": ["Run", "Walk", "Jump", "Sit"],   "correct": 1},
            {"id": 120, "type": "video",  "sign": "Run",    "options": ["Run", "Walk", "Jump", "Sit"],   "correct": 0},
        ],
        5: [
            {"id": 121, "type": "static", "sign": "Happy",  "options": ["Happy", "Sad", "Angry", "Scared"], "correct": 0},
            {"id": 122, "type": "video",  "sign": "Sad",    "options": ["Happy", "Sad", "Angry", "Scared"], "correct": 1},
            {"id": 123, "type": "static", "sign": "Angry",  "options": ["Happy", "Sad", "Angry", "Scared"], "correct": 2},
            {"id": 124, "type": "static", "sign": "Scared", "options": ["Happy", "Sad", "Angry", "Scared"], "correct": 3},
            {"id": 125, "type": "video",  "sign": "Excited","options": ["Bored", "Calm", "Excited", "Tired"],"correct": 2},
        ],
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    "hard": {
        1: [
            {"id": 201, "type": "camera", "sign": "Hello",       "options": None, "correct": None},
            {"id": 202, "type": "camera", "sign": "Thank You",   "options": None, "correct": None},
            {"id": 203, "type": "camera", "sign": "Please",      "options": None, "correct": None},
            {"id": 204, "type": "camera", "sign": "Sorry",       "options": None, "correct": None},
            {"id": 205, "type": "camera", "sign": "I Love You",  "options": None, "correct": None},
        ],
        2: [
            {"id": 206, "type": "camera", "sign": "Good morning",           "options": None, "correct": None},
            {"id": 207, "type": "camera", "sign": "Good night",             "options": None, "correct": None},
            {"id": 208, "type": "camera", "sign": "What is your name?",     "options": None, "correct": None},
            {"id": 209, "type": "camera", "sign": "My name is ...",         "options": None, "correct": None},
            {"id": 210, "type": "camera", "sign": "See you later",          "options": None, "correct": None},
        ],
        3: [
            {"id": 211, "type": "camera", "sign": "I love you",             "options": None, "correct": None},
            {"id": 212, "type": "camera", "sign": "I need help",            "options": None, "correct": None},
            {"id": 213, "type": "camera", "sign": "Can you hear me?",       "options": None, "correct": None},
            {"id": 214, "type": "camera", "sign": "Where is the bathroom?", "options": None, "correct": None},
            {"id": 215, "type": "camera", "sign": "I don't understand",     "options": None, "correct": None},
        ],
        4: [
            {"id": 216, "type": "camera", "sign": "Please repeat that",     "options": None, "correct": None},
            {"id": 217, "type": "camera", "sign": "I am hungry",            "options": None, "correct": None},
            {"id": 218, "type": "camera", "sign": "Call the doctor",        "options": None, "correct": None},
            {"id": 219, "type": "camera", "sign": "I am tired",             "options": None, "correct": None},
            {"id": 220, "type": "camera", "sign": "Stop please",            "options": None, "correct": None},
        ],
        5: [
            {"id": 221, "type": "camera", "sign": "How much does it cost?", "options": None, "correct": None},
            {"id": 222, "type": "camera", "sign": "I am learning sign language", "options": None, "correct": None},
            {"id": 223, "type": "camera", "sign": "Can we be friends?",     "options": None, "correct": None},
            {"id": 224, "type": "camera", "sign": "This is amazing",        "options": None, "correct": None},
            {"id": 225, "type": "camera", "sign": "Have a great day",       "options": None, "correct": None},
        ],
    },
}
