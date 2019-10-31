#python main.py --total_epochs 51 --batch_size 8 --model FlowNet2S --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingThingsCleanFlowTrain --training_dataset_root /data/FlyingThings3D --validation_dataset FlyingThingsCleanFlowValid --validation_dataset_root /data/FlyingThings3D -ng 1 > train.log 2>&1
#mv train.log work
#mv work work.fns.ftf
#python main.py --inference --model FlowNet2S --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --inference_dataset FlyingThingsCleanFlowValid --inference_dataset_root /data/FlyingThings3D -ng 1 --resume work.fns.ftf/FlowNet2S_model_best.pth.tar --save_flow > train.log 2>&1
# ./flow-code/color_flow work.fns.ftf/inference/run.epoch-0-flow-field/000100.flo 000100.png
#mv work/inference work.fns.ftf
#python main.py --total_epochs 51 --batch_size 8 --model FlowNet2S --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingThingsCleanDispTrain --training_dataset_root /data/FlyingThings3D --validation_dataset FlyingThingsCleanDispValid --validation_dataset_root /data/FlyingThings3D -ng 1 > train.log 2>&1
#mv train.log work
#mv work work.fns.ftd
#python main.py --inference --model FlowNet2S --loss=L1Loss1D --optimizer=Adam --optimizer_lr=1e-4 --inference_dataset FlyingThingsCleanDispValid --inference_dataset_root /data/FlyingThings3D -ng 1 --resume work.fns.ftd/FlowNet2S_model_best.pth.tar --save_flow > train.log 2>&1
# ./flow-code/color_flow work.fns.ftd/inference/run.epoch-0-flow-field/000100.flo 000100.png
#mv work/inference work.fns.ftd
#python main.py --total_epochs 51 --batch_size 8 --model FlowNet2S --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingThingsCleanContTrain --training_dataset_root /data/FlyingThings3D --validation_dataset FlyingThingsCleanContValid --validation_dataset_root /data/FlyingThings3D -ng 1 > train.log 2>&1
#mv train.log work
#mv work work.fns.ftc
#python main.py --inference --model FlowNet2S --loss=L1Loss1D --optimizer=Adam --optimizer_lr=1e-4 --inference_dataset FlyingThingsCleanContValid --inference_dataset_root /data/FlyingThings3D --resume work.fns.ftc/FlowNet2S_model_best.pth.tar --save_flow -ng 1 > train.log 2>&1
# ./flow-code/color_flow work.fns.ftc/inference/run.epoch-0-flow-field/000100.flo 000100.png
#mv work/inference work.fns.ftc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 8 > train.log 2>&1
mv train.log work
mv work work.q8.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 7 > train.log 2>&1
mv train.log work
mv work work.q7.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 6 > train.log 2>&1
mv train.log work
mv work work.q6.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 5 > train.log 2>&1
mv train.log work
mv work work.q5.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 4 > train.log 2>&1
mv train.log work
mv work work.q4.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 3 > train.log 2>&1
mv train.log work
mv work work.q3.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 2 > train.log 2>&1
mv train.log work
mv work work.q2.fc
python main.py --total_epochs 51 --batch_size 8 --model FlowNetQ --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset FlyingChairs --training_dataset_root /data/FlyingChairs/data/ --validation_dataset MpiSintelClean --validation_dataset_root /data/MPI-Sintel/training -ng 1 --bw 1 > train.log 2>&1
mv train.log work
mv work work.q1.fc
