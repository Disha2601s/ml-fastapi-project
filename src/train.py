from src.app import model as m
import os

if __name__ == '__main__':
    # expect data to exist at data/dataset.csv
    if not os.path.exists('data/dataset.csv'):
        print('data/dataset.csv not found, generating...')
        from src.data import generate_data as g
        g.__name__  # just run it

    metrics = m.train_and_save()
    clf, encoder = m.load_model()
    m.compute_slice_metrics(model=clf, encoder=encoder)
    print('Training complete. Metrics:', metrics)
