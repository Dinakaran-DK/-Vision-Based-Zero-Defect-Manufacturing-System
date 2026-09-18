import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_causal():
    df = pd.read_csv("telemetry_data.csv")
    def_df = df[df["has_defect"] == 1].copy()

    le_shift = LabelEncoder()
    def_df["shift_enc"] = le_shift.fit_transform(def_df["shift"])

    X = def_df[["machine_temp", "vibration", "humidity", "shift_enc"]]
    y = def_df["defect_type"]

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)

    joblib.dump(model, "causal_model.pkl")
    joblib.dump(le_shift, "shift_encoder.pkl")

if __name__ == "__main__":
    train_causal()