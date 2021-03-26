from utils.Dataset import Dataset
from AMVAE import MaeAEModel
from utils.print_result import print_result
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
if __name__ == '__main__':
    data = Dataset('handwritten_2views')
    x1, x2, gt = data.load_data()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    n_clusters = len(set(gt))

    act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    v1_aedims_ = [[240, 200],[200,240]]
    
    v2_aedims_ = [[216, 200],[200,216]]

    mae_dims_=[[200,150,32],[200,150,32],[32,150,200],[32,150,200]]
    dis_dims_=[200,150,1]
    para_lambda = 1
    batch_size = 100
    
    epochs = 50

    model=MaeAEModel(v1_aedims=v1_aedims_,v2_aedims=v2_aedims_,mae_dims=mae_dims_,dis_dims=dis_dims_)        
    H, gt = model.train_model(x1, x2, gt, epochs, batch_size)
    print_result(n_clusters, H, gt)
