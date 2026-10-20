from catboost import CatBoostClassifier
from sklearn.utils import Tags, ClassifierTags, TargetTags, InputTags

def __sklearn_tags__(self):
    return Tags(
        estimator_type="classifier",
        target_tags=TargetTags(required=True, one_d_labels=False, two_d_labels=False,
                                positive_only=False, multi_output=False, single_output=True),
        transformer_tags=None,
        classifier_tags=ClassifierTags(poor_score=False, multi_class=True, multi_label=False),
        regressor_tags=None,
        array_api_support=False,
        no_validation=False,
        non_deterministic=False,
        requires_fit=True,
        _skip_test=False,
        input_tags=InputTags(one_d_array=False, two_d_array=True, three_d_array=False,
                              sparse=False, categorical=False, string=False, dict=False,
                              positive_only=False, allow_nan=True, pairwise=False),
    )

CatBoostClassifier.__sklearn_tags__ = __sklearn_tags__
