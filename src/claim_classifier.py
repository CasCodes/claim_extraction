from transformers import pipeline

def load_classifier():
    classifier = pipeline(
        "text-classification", 
        model="Nithiwat/mdeberta-v3-base_claim-detection")
    return classifier

def predict_claims(sents, classifier):
    claims = []
    for sent in sents:
        pred = classifier(sent)[0]

        if pred["label"] == "LABEL_1":
            claims.append(sent)

    return claims