# train
python train.py --exp_path=pvraft --batch_size=2 --gpus=0,1 --num_epochs=20 --max_points=8192 --iters=8 --truncate_k=512 --corr_levels=3  --base_scales=0.25 --root=./
python train.py --refine --exp_path=pvraft_refine --batch_size=2 --gpus=4,5 --num_epochs=10 --max_points=8192 --iters=32 --truncate_k=512 --corr_levels=3 --base_scales=0.25 --root=./ --weights=pvraft
# test
python test.py --dataset=FT3D --exp_path=pvraft --gpus=4 --max_points=8192 --iters=8 --truncate_k=512 --corr_levels=3 --base_scales=0.25 --root=./ --weights=./experiments/pvraft/checkpoints/best_checkpoint.params
python test.py --dataset=KITTI --exp_path=pvraft --gpus=4 --max_points=8192 --iters=8 --truncate_k=512 --corr_levels=3 --base_scales=0.25 --root=./ --weights=./experiments/pvraft/checkpoints/best_checkpoint.params
# test refine
python test.py --refine --dataset=FT3D --exp_path=pvraft_refine --gpus=4 --max_points=8192 --iters=32 --truncate_k=512 --corr_levels=3 --base_scales=0.25 --root=./ --weights=./experiments/pvraft_refine/checkpoints/best_checkpoint.params
python test.py --refine --dataset=KITTI --exp_path=pvraft_refine --gpus=4 --max_points=8192 --iters=32 --truncate_k=512 --corr_levels=3 --base_scales=0.25 --root=./ --weights=./experiments/pvraft_refine/checkpoints/best_checkpoint.params