best setting for multimnist is 
    - 50 epochs
    - anneal-kl (5e-5, 1e-4, 5e-4, 1e-3) every 10 epochs
    - 1 for lambda_x, lambda_xy, lambda_yx
    - 100 for lambda_y since text needs more help
    - learning rate 1e-3, no annealing

best setting for scramblemnist is
    - 50 epochs
    - anneal-kl (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1) every 5 epochs
    - 1 for lambda_x, lambda_y, lambda_xy, lambda_yx
    - learning rate 1e-3, no annealing

