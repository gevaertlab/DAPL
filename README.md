# DAPL: A deep learning framework for imputing missing values in genomic data

Yeping Lina Qiu, Hong Zheng, Olivier Gevaert
Stanford University

This implemented a denoising autoencoder with partial loss (DAPL) as a deep learning based alternative for imputating missing values especially for large datasets, that achieves comparable or better performance than conventional methods with less computational burden.

https://www.biorxiv.org/content/early/2018/09/03/406066

To run prediction script with trained model and sample test dataset:

python predict_random_missing.py input_name output_name model_path feature_size nonmissing_percent     
e.g., python predict_random_missing.py 'testdata_100sample.csv' 'testloss_100sample.csv' 'imputationmodel.ckpt' 17176 0.7

To run training script:

python train_random_missing.py input_name output_path feature_size nonmissing_perc batchsize learning_rate num_epochs