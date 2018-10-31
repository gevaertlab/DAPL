import os
import tensorflow as tf
import pandas as pd
import time
from autoencoder import *
import random
import numpy as np

        
def reconstruct_loss(dataset_test_uncorrutped,dataset_test,autoencoder_fun,checkpoint_file='default.ckpt',missing_ind=None):
    input_image, reconstructed_image = autoencoder_fun(batch_shape)

    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as session:

            print(("Loading variables from '%s'." % checkpoint_file))
            saver.restore(session, checkpoint_file)
            print('restored')    
            dataset_size = dataset_test.shape[0]
            print("Dataset size:", dataset_size)            
            
            dataset_test=np.asarray(dataset_test).astype("float32") 
            dataset_test_uncorrutped=np.asarray(dataset_test_uncorrutped).astype("float32") 
             
            reconstruct= session.run(reconstructed_image, feed_dict={input_image: dataset_test})        
            loss=rmse_loss(reconstruct,dataset_test_uncorrutped,missing_ind)
            

    return(loss)

def rmse_loss(reconstructed, original,missing_ind):
    rmse = np.sqrt(((reconstructed[0,missing_ind] - original[0,missing_ind]) ** 2).mean())
    return rmse


def mask_dfrow(row,perc):
    sample=np.random.binomial(1,perc,size=row.size)
    corrupted=row*sample
    return(corrupted)


if __name__ == '__main__':
    
        df= pd.read_csv("rna_naremoved_logtransformed_normalized.csv")  
        df.drop(df.columns[[0]], axis=1, inplace=True)
        arr=list(range(df.shape[0]))
        random.seed(1)
        random.shuffle(arr)
        use_ind=arr[0:int(df.shape[0]*0.75)]
        holdout_ind=arr[int(df.shape[0]*0.75):len(arr)]
        df_use = df.iloc[use_ind]
        df_holdout = df.iloc[holdout_ind]
        test_sample_n=100
        feature_size=17176
        holdout_cohort=df_holdout.iloc[0:test_sample_n]
        
        nonmissing_perc=0.7
        np.random.seed(1)
        corrupted_holdout_cohort=holdout_cohort.apply(mask_dfrow,perc=nonmissing_perc,axis=1)
        
        loss_list=0
        for i in range(0,corrupted_holdout_cohort.shape[0]):
            cur_test=corrupted_holdout_cohort.iloc[i:i+1,:]
            true_cur_test=holdout_cohort.iloc[i:i+1,:]
        
            missing_index=np.where(cur_test.iloc[0,:]==0)[0]
            batch_shape = (1, feature_size)
            np.set_printoptions(threshold=np.inf)
            tf.reset_default_graph()
            loss_val=reconstruct_loss(true_cur_test,cur_test,autoencoder4_d, 'imputationmodel.ckpt',missing_index)
            
            loss_list=np.append(loss_list,loss_val)  
            print(loss_val)
            if i%5==0:
                np.savetxt("testloss100sample.csv", loss_list, delimiter="\t")
        
        
                
                
