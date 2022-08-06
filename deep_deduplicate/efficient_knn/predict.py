import typing as t

import pandas as pd
from pandas import DataFrame

from efficient_knn import __version__ as _version
from efficient_knn.config.core import config
from efficient_knn.pipeline import text_pipeline, image_pipeline
from efficient_knn.processing.combine_predictions import CombinePredictionsTransform
from efficient_knn.processing.data_manager import load_pipeline, load_dataset
from efficient_knn.processing.index_transform import IndexTransform



def make_prediction(
    input_data: t.Union[pd.DataFrame, dict],
) -> DataFrame:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    # validated_data, errors = validate_inputs(input_data=data)
    # results = {"predictions": None, "version": _version, "errors": errors}
    #
    # if not errors:
    #     predictions = _price_pipe.predict(
    #         X=validated_data[config.model_config.features]
    #     )
    #     results = {
    #         "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
    #         "version": _version,
    #         "errors": errors,
    #     }

    text_pipeline.fit(data, data["posting_id"])
    image_pipeline.fit(data, data["posting_id"])

    text_pred = text_pipeline.predict(data)
    image_pred = image_pipeline.predict(data)

    idx_transform = IndexTransform()
    data["image_pred"] = idx_transform.fit_transform(image_pred, data["posting_id"])
    data["text_pred"] = idx_transform.fit_transform(text_pred, data["posting_id"])

    combine_transform = CombinePredictionsTransform()
    combine_transform.fit_transform(data)

    response = pd.DataFrame()

    response['posting_id'] = data['posting_id']
    response['matches'] = data['pred_combined']

    return response

if __name__ == "__main__":
    input = load_dataset("train.csv")
    make_prediction(input)
