# AI-Based Resume Screening & Candidate Ranking (Java + NLP)

No APIs, no external libraries. Pure JDK 11+.

## Run
    cd ResumeScreener
    javac -d out src/*.java
    java -cp out Server
Open http://localhost:8080

## How it works
1. Preprocessor: lowercase, clean, tokenize, stop-word removal, stemming.
2. NaiveBayes: multinomial Naive Bayes trained from scratch on data/training.csv (format `Category|resume text`). It predicts the job category of the JD and the role-fit probability of each resume.
3. TF-IDF + cosine similarity between JD and each resume.
4. Skill extraction using data/skills.txt (matched / missing skills).
5. Final score = 0.5 x TF-IDF cosine + 0.3 x skill match + 0.2 x Naive Bayes role fit.

## Improve it
- Replace data/training.csv with the Kaggle "Resume Dataset" (UpdatedResumeDataSet.csv, ~960 resumes, 25 categories) converted to `Category|text` format.
- Add PDF upload using Apache PDFBox.
- Add accuracy evaluation (80/20 split, confusion matrix) for your report.
