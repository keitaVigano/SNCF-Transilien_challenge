from sklearn.pipeline import Pipeline
import category_encoders as ce

def build_pipeline(model):
    return Pipeline([
        ('target_enc', ce.TargetEncoder(
            cols=['gare', 'arret', 'quarter', 'gare_arret'],
            smoothing=10
        )),
        ('model', model)])

