import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import handle_missing_value, apply_numeric_scaling, save_model, apply_label_encoder, apply_one_hot_encoder


class ModelTrainer:
    def __init__(self):
        self.genre_id_ohe_model_path = 'models/genre_id_ohe.pkl'
        self.vertical_id_ohe_model_path = 'models/vertical_id_ohe.pkl'
        self.gender_le_model_path = 'models/gender_le.pkl'
        self.gender_ohe_model_path = 'models/gender_ohe.pkl'
        self.scaled_model_path = 'models/scaled.pkl'
        self.cluster_model1_model_path = 'models/cluster_model1.pkl'
        self.cluster_model2_model_path = 'models/cluster_model2.pkl'
        self.classification_model_path = 'models/classification.pkl'
        self.user_cluster_ref_path = 'models/user_cluster_ref.csv'
        self.df_writers = pd.read_excel('data/writers.xlsx', engine='openpyxl', usecols='B:AA')
        print('Original Writer shape: ', self.df_writers.shape)
        self.df_writer_vector_feats = None
        self.df_writer_train = None
        # Use only specific columns
        self.df_writer_feats = self.df_writers[['genre_id', 'vertical_id', 'gender_cat', 'word_count_per_day']]

    def data_preprocessing(self):
        print('Missing value counts for writer data: ', self.df_writers.isnull().sum())

        # genre_id
        handle_missing_value(self.df_writer_feats, 'genre_id')
        self.df_writer_feats['genre_id'] = self.df_writer_feats['genre_id'].astype('int')
        print('unique genre_id: ', list(set(self.df_writer_feats['genre_id'])))
        genre_id_array = apply_one_hot_encoder(self.df_writer_feats, 'genre_id', self.genre_id_ohe_model_path)

        # vertical_id
        handle_missing_value(self.df_writer_feats, 'vertical_id')
        self.df_writer_feats['vertical_id'] = self.df_writer_feats['vertical_id'].astype('int')
        print('unique vertical_id: ', list(set(self.df_writer_feats['vertical_id'])))
        vertical_id_array = apply_one_hot_encoder(self.df_writer_feats, 'vertical_id', self.vertical_id_ohe_model_path)

        # gender
        self.df_writer_feats.loc[
            (~self.df_writer_feats['gender_cat'].isin(['male', 'female'])), 'gender_cat'] = 'others'
        self.df_writer_feats['gender_cat'] = apply_label_encoder(self.df_writer_feats, 'gender_cat',
                                                                 self.gender_le_model_path)
        gender_array = apply_one_hot_encoder(self.df_writer_feats, 'gender_cat', self.gender_ohe_model_path)

        # word count
        word_count_median = self.df_writer_feats['word_count_per_day'].median()
        print('Word count median: ', int(word_count_median))
        handle_missing_value(self.df_writer_feats, 'word_count_per_day', value=word_count_median)
        self.df_writer_feats['word_count_per_day'] = self.df_writer_feats['word_count_per_day'].astype('int')

        # concat all
        self.df_writer_vector_feats = pd.concat([
            pd.DataFrame(genre_id_array),
            pd.DataFrame(vertical_id_array),
            pd.DataFrame(gender_array),
            pd.DataFrame(self.df_writer_feats['word_count_per_day'])],
            axis=1
        )
        print('Features shape: ', self.df_writer_vector_feats.shape)

    def build_multiple_clusters(self):
        self.df_writer_vector_feats = apply_numeric_scaling(self.df_writer_vector_feats, self.scaled_model_path)
        df_scaled_features = pd.DataFrame(self.df_writer_vector_feats)

        # Clustering model - 1
        model_1 = DBSCAN(eps=5, min_samples=3)
        pred_y = model_1.fit_predict(df_scaled_features)
        save_model(model_1, self.cluster_model1_model_path)

        df_scaled_features['pred_y'] = pred_y

        df_rem_scaled_features = df_scaled_features[df_scaled_features['pred_y'] == -1]
        max_pred_y = max(pred_y)

        # Clustering model - 2
        model_2 = DBSCAN(eps=10, min_samples=3)
        save_model(model_2, self.cluster_model2_model_path)
        pred_y = model_2.fit_predict(
            df_rem_scaled_features[df_rem_scaled_features.columns[~df_rem_scaled_features.columns.isin(['pred_y'])]]
        )
        pred_y = pred_y + (max_pred_y + 1)  # add + max
        pred_y[pred_y == min(pred_y)] = -1  # set -1 again

        df_scaled_features.loc[list(df_rem_scaled_features.index), 'pred_y'] = pred_y  # update

        # rename 'pred_y' to 'cluster'
        df_scaled_features['cluster'] = df_scaled_features['pred_y']
        df_scaled_features.drop(['pred_y'], axis=1, inplace=True)
        print('Sacled features shape: ', df_scaled_features.shape)
        print('\nClustering done..')
        return df_scaled_features

    def build_classification(self, df_scaled_features):
        clf = RandomForestClassifier()
        X = df_scaled_features[df_scaled_features.columns[~df_scaled_features.columns.isin(['cluster'])]]
        y = df_scaled_features['cluster']
        clf.fit(X, y)
        y_pred = clf.predict(X)
        print('Accuracy: ', accuracy_score(y, y_pred))

        # save user_id with cluster number for future reference
        df_scaled_features['user_id'] = self.df_writers['user_id']
        df_scaled_features.to_csv(self.user_cluster_ref_path, index=False)

        save_model(clf, self.classification_model_path)
        print('\nClassification done..')

    def run(self):
        self.data_preprocessing()
        df_scaled_features = self.build_multiple_clusters()
        self.build_classification(df_scaled_features)


if __name__ == '__main__':
    ModelTrainer().run()
