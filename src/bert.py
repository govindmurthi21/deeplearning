def run():
    # train_spam_no_spam_classifier()
    # predict_spam_or_no_spam()
    simple_qan()

def train_spam_no_spam_classifier():
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    import tensorflow as tf

    df = pd.read_csv(Path("data", "SMSSpamCollection.csv").resolve(), sep='\t', names=['label', 'message'])
    print(df.columns.tolist())
    x = list(df['message'])
    y = list(df['label'])
    y = list(pd.get_dummies(y, drop_first=True,dtype=int)['spam'])
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings  = tokenizer(x_train, truncation=True, padding=True)
    test_encodings  = tokenizer(x_test, truncation=True, padding=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
    model:TFDistilBertForSequenceClassification = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
    model.evaluate(test_dataset.shuffle(len(x_test)).batch(16), return_dict=True, batch_size=16)
    model.save_pretrained(Path("pretrained", "models", "spam_no_spam_model").resolve())
    tokenizer.save_pretrained(Path("pretrained", "tokenizers", "spam_no_spam_tokenizer").resolve())

def predict_spam_or_no_spam(): 
    from pathlib import Path
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    import tensorflow as tf
    import numpy as np

    model_path = Path("pretrained", "models", "spam_no_spam_model").resolve()
    tokenizer_path = Path("pretrained", "tokenizers", "spam_no_spam_tokenizer").resolve()
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    string_list = ['	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call T']
    encodings  = tokenizer(string_list, max_length=512, truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    print(len(dataset))
    preds = model.predict(dataset.batch(1))
    result = tf.nn.softmax(preds.logits, axis=1).numpy()[0]
    if result[1] > result[0] and result[1] >= 0.75:
        print("This message was classified as spam")
    else:
        print("This result was not a spam")

    print(result)

def simple_qan():
    from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline
    context = (
        """Black Knight (NYSE:BKI) is a leading provider of integrated software, data and analytics solutions that facilitate and automate many of the business processes across the homeownership life cycle. 
        Black Knight is committed to being a premier business partner that clients rely on to achieve their strategic goals, realize greater success and better serve their customers by delivering best-in-class software, services and insights with a relentless commitment to excellence, innovation, integrity and leadership.
        Its mission is to be the PREMIER PROVIDER of software, data and analytics, known for CLIENT FOCUS AND PRODUCT EXCELLENCE; and to deliver INNOVATIVE, seamlessly INTEGRATED solutions with URGENCY.
       Anthony Jabbour is responsible for providing leadership and direction to the company’s management and Board of Directors.
       Before being appointed Executive Chairman, Anthony served as Chairman and CEO of Black Knight, where he helped substantially increase organic growth, led the company to deliver numerous digital solutions and other innovative capabilities, and oversaw nine acquisitions to provide greater shareholder value and help transform the industries Black Knight serves.
       """
    )
    questions = [
        'Who is the ceo of black knight?',
        "What is black knight known for?",
        'What does black knight do?'
    ] 
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    result = nlp({
        'question': questions[1],
        'context': context
    })
    print(result['answer'])

