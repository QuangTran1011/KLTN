import json

import mlflow
import torch


class SkipGramInferenceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        """
        This load_context method is automatically called when later we load the model.
        """
        json_path = context.artifacts["id_mapping"]
        with open(json_path, "r") as f:
            self.id_mapping = json.load(f)

    def predict(self, context, model_input, params=None):
        """
        Args:
            context: The context object in mlflow.pyfunc often contains pointers to artifacts that are logged alongside the model during training (like feature encoders, embeddings, etc.)
        """
        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        item_1_indices = [
            self.id_mapping["item_to_index"].get(id_) for id_ in model_input["item_1_ids"]
        ]
        item_2_indices = [
            self.id_mapping["item_to_index"].get(id_) for id_ in model_input["item_2_ids"]
        ]
        infer_output = self.infer(item_1_indices, item_2_indices).tolist()
        return {
            "item_1_ids": model_input["item_1_ids"],
            "item_2_ids": model_input["item_2_ids"],
            "scores": infer_output,
        }

    def infer(self, item_1_indices, item_2_indices):
        item_1_indices = torch.tensor(item_1_indices)
        item_2_indices = torch.tensor(item_2_indices)
        output = self.model(item_1_indices, item_2_indices)
        return output.detach().numpy()
