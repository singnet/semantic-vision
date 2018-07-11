This model was cloned from https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA
HyperNet was added instead of point-wise product. Results are in the folder "saved_models".
Architecture of HyperNet is following:
q_vector (question embedding) -> N_neirons -> [control_mat_size x control_mat_size]

v_vector (visual features) -> control_mat_size -> applying of control mat -> 1280 neirons

log_100.txt and log_300.txt means we've used 100 and 300 accordingly for N_neirons. Also, in our case control_mat_size = N_neirons. 
