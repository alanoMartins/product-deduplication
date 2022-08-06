from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from efficient_knn import config
from efficient_knn.processing.efficient_feature_extractor import ImageFeatureExtractor
from efficient_knn.processing.estimator import DistanceEstimator
from efficient_knn.processing.features import SanitizeText

text_pipeline = Pipeline(
    [
        (
            "text_clean",
            SanitizeText(variables=config.model_config.text_var),
        ),
        (
            "text_vectorize",
            TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english'),
        ),
        (
            "knn_distance",
            DistanceEstimator(threshold=0.25),
        )
    ]
)


image_pipeline = Pipeline(
    [
        (
            "image_features",
            ImageFeatureExtractor(variables=config.model_config.image_var,
                                  model_path=config.model_config.weight_path,
                                  image_path=config.model_config.image_path),
        ),
        (
            "knn_distance",
            DistanceEstimator(threshold=0.15),
        )
    ]
)
