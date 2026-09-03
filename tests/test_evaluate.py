import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import create_explainer, generate_summary_plot


@pytest.fixture
def tiny_model_and_data():
    X = pd.DataFrame(
        np.random.RandomState(0).rand(40, 3),
        columns=["f1", "f2", "f3"],
    )
    y = (X["f1"] + X["f2"] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    return model, X


def test_create_explainer_does_not_write_any_files(tiny_model_and_data, tmp_path):
    model, X = tiny_model_and_data
    before = set(os.listdir(tmp_path))

    explainer = create_explainer(model)

    after = set(os.listdir(tmp_path))
    assert before == after, "create_explainer() must be pure -- no side-effect file writes"
    assert explainer is not None


def test_generate_summary_plot_writes_expected_file(tiny_model_and_data, tmp_path):
    model, X = tiny_model_and_data
    explainer = create_explainer(model)

    output_dir = str(tmp_path / "outputs")
    plot_path = generate_summary_plot(explainer, X, output_dir=output_dir, sample_size=10)

    assert os.path.exists(plot_path)
    assert plot_path == os.path.join(output_dir, "shap_summary.png")
