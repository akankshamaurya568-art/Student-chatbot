from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

intents = {

    "greeting": {
        "patterns": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "namaste"
        ],
        "responses": [
            "Hello! How can I help you?",
            "Hi! Welcome to Student Support Services.",
            "Hello! What information do you need?"
        ]
    },

    "admission": {
        "patterns": [
            "admission",
            "how to take admission",
            "admission process",
            "college admission",
            "how can I apply",
            "admission details"
        ],
        "responses": [
            "You can get admission information from the college admission office or official website.",
            "For admission details, please check the latest admission notice."
        ]
    },

    "fees": {
        "patterns": [
            "fees",
            "college fees",
            "fee structure",
            "semester fees",
            "how much is the fee",
            "fee details"
        ],
        "responses": [
            "Please contact the college accounts office for the latest fee structure.",
            "The latest fee details are available from the college administration."
        ]
    },

    "scholarship": {
        "patterns": [
            "scholarship",
            "scholarship details",
            "how to apply scholarship",
            "scholarship form",
            "scholarship information",
            "government scholarship"
        ],
        "responses": [
            "You can apply for eligible scholarships through the official scholarship portal.",
            "For scholarship-related help, contact the college scholarship section."
        ]
    },

    "exam": {
        "patterns": [
            "exam",
            "exam date",
            "semester exam",
            "when is exam",
            "exam timetable",
            "examination"
        ],
        "responses": [
            "Please check the latest examination timetable issued by your university or college.",
            "For exam-related information, contact the examination department."
        ]
    },

    "result": {
        "patterns": [
            "result",
            "exam result",
            "semester result",
            "result date",
            "check result",
            "result information"
        ],
        "responses": [
            "You can check your result on the official university result portal.",
            "For result-related issues, contact the examination department."
        ]
    },

    "attendance": {
        "patterns": [
            "attendance",
            "attendance percentage",
            "my attendance",
            "attendance requirement",
            "attendance rules"
        ],
        "responses": [
            "You can check your attendance through the college attendance system or contact your department."
        ]
    },

    "hostel": {
        "patterns": [
            "hostel",
            "hostel facility",
            "hostel admission",
            "hostel fees",
            "hostel information",
            "hostel available"
        ],
        "responses": [
            "For hostel availability, fees and admission, please contact the hostel administration.",
            "The hostel office can provide the latest hostel information."
        ]
    },

    "library": {
        "patterns": [
            "library",
            "library timing",
            "library books",
            "library facility",
            "book issue",
            "library information"
        ],
        "responses": [
            "The library provides books and study resources for students.",
            "For library timings and book issues, please contact the library desk."
        ]
    },

    "timetable":{
"patterns": [
            "timetable",
            "class timetable",
            "time table",
            "lecture schedule",
            "class schedule"
        ],
        "responses": [
            "Please check your department notice board or student portal for the latest timetable."
        ]
    },

    "placement": {
        "patterns": [
            "placement",
            "campus placement",
            "placement cell",
            "job placement",
            "placement information",
            "companies for placement"
        ],
        "responses": [
            "The placement cell provides information about campus recruitment and companies.",
            "Please contact the Training and Placement Cell for the latest placement information."
        ]
    },

    "internship": {
        "patterns": [
            "internship",
            "internship opportunities",
            "internship information",
            "internship help",
            "how to get internship"
        ],
        "responses": [
            "The Training and Placement Cell can provide information about internship opportunities.",
            "You can also check internship opportunities on verified company and career portals."
        ]
    },

    "contact": {
        "patterns": [
            "contact",
            "college contact",
            "contact number",
            "office number",
            "college phone number"
        ],
        "responses": [
            "Please check your college's official website for the latest contact details."
        ]
    },

    "goodbye": {
        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "thank you bye",
            "exit"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "Thank you for using Student Support Services.",
            "See you later!"
        ]
    }
}

patterns = []
tags = []

for tag, data in intents.items():
    for pattern in data["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(patterns)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, tags)

def chatbot(user_input):

    user_vector = vectorizer.transform([user_input])

    prediction = model.predict(user_vector)[0]

    probabilities = model.predict_proba(user_vector)[0]

    confidence = max(probabilities)

    # If chatbot is not confident
    if confidence < 0.15:
        return (
            "Sorry, I couldn't understand your question. "
            "Please ask about admission, fees, scholarship, "
            "exam, result, hostel, library, placement, etc."
        )

    return random.choice(intents[prediction]["responses"])

print("\n====================================")
print("   STUDENT SUPPORT SERVICES CHATBOT")
print("====================================")
print("Type 'bye' or 'exit' to close the chatbot.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() in ["bye", "exit", "quit"]:
        print("Bot: Goodbye! Have a great day!")
        break

    response = chatbot(user_input)

    print("Bot:", response)