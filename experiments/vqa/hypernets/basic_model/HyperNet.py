import torch

device = torch.device("cpu:0")
device2 = torch.device("cuda:0")

def HyperNet(q_arr, v_arr):
    q_size = list(q_arr.size())
    v_size = list(v_arr.size())
    
    hyper_model = torch.nn.Sequential(
        torch.nn.Linear(q_size[1], 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, v_size[1] * v_size[1])
    )
    
    q_arr = q_arr.new_tensor(q_arr.data, dtype=torch.float32, device=device)
    qv_mat = hyper_model(q_arr)
    
    qv_mat = qv_mat.reshape((-1, v_size[1], v_size[1]))
    
    qv_mat = qv_mat.new_tensor(qv_mat.data, dtype=torch.float32, device=device2)
    
    result = torch.einsum('bj,bjk->bk', (v_arr, qv_mat))
    output_relu_layer = torch.nn.ReLU()
    result = output_relu_layer(result)
    return result
