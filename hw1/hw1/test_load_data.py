import load_data

def test_sample_batch():
    d = load_data.DataGenerator(3, 5) # 3 classes, 5 samples per class

    d.sample_batch("train", 1)