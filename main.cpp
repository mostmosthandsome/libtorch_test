#include "torch/script.h"
#include "Eigen/Dense"
int obs_dim = 49,obs_seq_len = 50;
Eigen::VectorXf obs(49),obs_history(2450);

int main()
{
    int obs_dim = 49,obs_seq_len = 50;
    Eigen::VectorXf obs(49),obs_history(2450);

    obs << 1.43188e-08, -1.19241e-08, -7.79237e-10, -4.39196e-10, 5.51574e-12, -1, 0.578147,
    8.63105e-10, 7.4644e-10, 0, 0, 1.02582e-09, 0, 0, 0, 0, 0, 5.99261e-10, 0, 0, 0,
    7.46807e-09, -3.00407e-09, 1.76177e-09, -4.82405e-09, 4.83885e-09, -2.67542e-10,
    -1.06283e-09, -3.26135e-09, 7.69607e-10, 4.91082e-09, -4.83495e-09, -5.01063e-10,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1.83697e-16, -1, -1.83697e-16;

    obs_history.setZero();
    for(int i =0 ; i < 49; ++i)    obs_history[2401 + i] = obs[i]; 
    torch::jit::Module model = torch::jit::load("policy.pt");

    torch::Tensor obs_history_tensor = torch::from_blob(obs_history.data(), {1, obs_dim * obs_seq_len});
    torch::Tensor acTorch = model({obs_history_tensor}).toTensor();
    //将actorch转为VectorXf
    Eigen::Matrix<float, 12,1> tVec{acTorch.data_ptr<float>()}; 
    std::cout << tVec.col(0).transpose();
}