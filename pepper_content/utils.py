import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def save_model(model, model_path):
    fp = open(model_path, 'wb')
    pickle.dump(model, fp)
    fp.close()

def load_model(model_path):
    fp = open(model_path, 'rb')
    model = pickle.load(fp)
    fp.close()
    return model


def apply_label_encoder(temp_df, column, model_path):
    le = LabelEncoder()
    res = le.fit_transform(temp_df[column])
    save_model(le, model_path)
    return res

def apply_one_hot_encoder(temp_df, column, model_path):
    ohe = OneHotEncoder()
    res = ohe.fit_transform(temp_df[[column]]).toarray()
    save_model(ohe, model_path)
    return res

def apply_numeric_scaling(temp_df, model_path):
    scaler = StandardScaler()
    res = scaler.fit_transform(temp_df)
    save_model(scaler, model_path)
    return res

def do_transform_using_label_encoder(temp_df, column, model_path):
    le = load_model(model_path)
    return le.transform(temp_df[column])

def do_transform_using_one_hot_encoder(temp_df, column, model_path):
    ohe = load_model(model_path)
    return ohe.transform(temp_df[[column]]).toarray()

def handle_missing_value(temp_df, column, value=0):
    temp_df[column].fillna(value, inplace=True)

def do_transform_numeric_scaling(temp_df, model_path):
    scaler = load_model(model_path)
    return scaler.transform(temp_df)

def predict_from_cluster(temp_df):
    model = load_model(model_path='models/cluster_model1.pkl')
    return model.predict(temp_df)


