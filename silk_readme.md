1. Install all `code2vec` requirements
2. Download trained model from step 2
3. Download code dataset from here and extract it: http://groups.inf.ed.ac.uk/cup/codeattention/
4. Run `python silk_preprocess.py --load models/java14_model/saved_model_iter8.release` and enter the path to the directory of code files to generate dataset for (include '/' at the end)
5. Wait for `silk_dataset.txt` to be generated
