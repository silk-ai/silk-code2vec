from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
from extractor import Extractor
from os import listdir
from time import time

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import Code2VecModel
    return Code2VecModel(config)


if __name__ == '__main__':
    start_time = time()
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_model_dynamically(config)
    config.log('Done creating code2vec model')

    model.predict([])

    # predictor = InteractivePredictor(config, model)
    # predictor.predict()

    code_directory_path = input('Enter path to directory to use to generate dataset: ')
    dataset = []
    path_extractor = Extractor(config,
        jar_path=JAR_PATH,
        max_path_length=MAX_PATH_LENGTH,
        max_path_width=MAX_PATH_WIDTH)

    filenames = [filename for filename in listdir(code_directory_path) if filename.endswith('.java')]
    print(filenames[:10])
    total_files = len(filenames)

    for i, filename in enumerate(filenames):
        print('{}/{}'.format(i, total_files))
        try:
            predict_lines, hash_to_string_dict = path_extractor.extract_paths(code_directory_path + filename)
        except ValueError as e:
            print(e)
            continue

        raw_prediction_results = model.predict(predict_lines)
        for function_result in raw_prediction_results:
            dataset.append(function_result.topk_predicted_words[0])


    with open('silk_dataset.txt', 'w') as outfile:
        try:
            outfile.write(','.join(dataset))
        except TypeError as e:
            print(e)
            print(dataset)

    model.close_session()
    end_time = time()
    print('Total time: {}'.format(end_time - start_time))
