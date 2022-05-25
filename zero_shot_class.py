from transformers import pipeline

classifier = pipeline("zero-shot-classification")

classifier("shine bright like a diamond Find light in the beautiful sea, I choose to be happy You and I, you and I, we're like diamonds in the sky You're a shooting star I see, a vision of ecstasy When you hold me, I'm alive We're like diamonds in the sky I knew", candidate_labels=["happy","cheerful","euphoric","sad","depressed", "angry"],)

