
import pandas as pd
import os
from hazm import Normalizer, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report


base_path = os.path.join(os.path.dirname(__file__), 'data/')


def data_prepartion(in_file, out_file, type):
    file = open(in_file, "r", encoding='utf-8')
    file_wrt = open(out_file, "w", encoding='utf-8')
    rows = []
    lines = ''
    for line in file.readlines():
        line = line.strip()
        if len(rows) == 0:
            file_wrt.write(line+'\n')
            rows.append(line)
            continue
        lines += line + ' '
        if type=='test' and (line.endswith('nan') or line.endswith(']"') or line.endswith('[]')):
            file_wrt.write(lines + '\n')
            rows.append(lines)
            lines = ''
        elif type=='train' and (line.endswith('recommended')):
            file_wrt.write(lines+ '\n')
            rows.append(lines)
            lines = ''
    file_wrt.close()
    return rows


def merge_data(src_path, tgt_path, type):
    src = pd.read_csv(src_path)
    src['title'] = src['title'].fillna('')
    src['comment'] = src['comment'].fillna('')
    src["text"] = src["title"] + src["comment"]
    if type == 'test':
        tgt = pd.read_csv(tgt_path)
        # print(tgt['recommend'].value_counts())
        src = pd.merge(src, tgt, on=["id"], how="inner")
    return src


def text_normalizer(df):
    df['recommend'] = df['recommend'].map({'recommended': 1, 'not_recommended': 0,
                                   'recommended ':1, 'not_recommended ':0})
    normalizer = Normalizer()
    df["text"] = df["text"].str.replace('_x000D_', ' ').str.replace(r'[^\w\s]+', '')
    df['text_normalized'] = df["text"].apply(lambda txt: normalizer.normalize(txt))
    return df


def train_model(training_set, validation_set):
    stop_words = open(base_path+'stopwords_fa.txt', encoding='utf-8').readlines()
    model = Pipeline([("vect", TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1,1), stop_words=stop_words)),
                      ("tfidf", TfidfTransformer()),
                      ("clf", ML()), ])

    # parameters = { 'clf__max_iter': [2000,2500,3000,3500,4000,4500,5000]}
    # gs_clf = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
    # gs_clf = gs_clf.fit(training_set.text_normalized[:5000], training_set.recommend[:5000])
    # print(gs_clf.best_params_)
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # RandomForestClassifier(n_estimators=3, bootstrap = True, max_features = 'sqrt', criterion='entropy')
    # MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.005, max_iter=20))
    model.fit(training_set.text_normalized[:5000], training_set.recommend[:5000])
    preds = model.predict(validation_set.text_normalized)
    file=open('result.csv', 'w', encoding='utf-8')
    file.write('id,recommend\n')
    for i in range(len(preds)):
        file.write('{},{}\n'.format(i, 'recommended' if preds[i]==1 else 'not_recommended')
    file.close()
    accuracy = accuracy_score(validation_set.recommend, preds)
    f1 = f1_score(validation_set.recommend, preds)
    print(classification_report(validation_set.recommend, preds))
    print("Accuracy = {}\nF1 = {}".format(accuracy, f1))


if __name__ == '__main__':
    need_fileWrite = False
    train_init_file = base_path + 'train.csv'
    train_clear_file = base_path + 'CI_train_clear.csv'
    test_init_file = base_path + 'CI_test.csv'
    test_clear_file = base_path + 'CI_test_clear.csv'
    test_target_file = base_path + 'out_sample_test.csv'
    if need_fileWrite:
        data_prepartion(train_init_file, train_clear_file, 'train')

    train_data = merge_data(train_clear_file, '', 'train')
    train_data = text_normalizer(train_data)

    # train_data, val_data = train_test_split(train_data[['text_normalized', 'recommend']], test_size=0.1, random_state=0)
    # train_model(train_data, val_data)
    test_data = merge_data(test_clear_file, test_target_file, 'test')
    test_data = text_normalizer(test_data)
    train_model(train_data[['text_normalized', 'recommend']], test_data[['id', 'text_normalized', 'recommend']])
    # print(data.head())