from app.services.search import normalize_question

def test_normalize_question():
    original = "what does this meaning of colossians 3?"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", result)

test_normalize_question()