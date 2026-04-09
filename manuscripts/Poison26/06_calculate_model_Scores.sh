SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif

WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25
mkdir -p $WORKINGDIR/MNIST_stats

MERGE=$WORKINGDIR/bin/infer/merge_MNIST-CSV.py
python $MERGE $WORKINGDIR/MNIST_test_model1_ALL/ --output_file $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv

INFER=$WORKINGDIR/bin/infer/infer_MNIST.py

# Model file
MODEL=MNIST_model1.pt
singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER --model /models/$MODEL --csv $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv --outfile $WORKINGDIR/MNIST_stats/Model1-Poison_Model1-Performance.tsv
MODEL=MNIST_model2.pt
singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER --model /models/$MODEL --csv $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv --outfile $WORKINGDIR/MNIST_stats/Model1-Poison_Model2-Performance.tsv
MODEL=MNIST_model3.pt
singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER --model /models/$MODEL --csv $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv --outfile $WORKINGDIR/MNIST_stats/Model1-Poison_Model3-Performance.tsv
MODEL=MNIST_model4.pt
singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER --model /models/$MODEL --csv $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv --outfile $WORKINGDIR/MNIST_stats/Model1-Poison_Model4-Performance.tsv
MODEL=MNIST_model5.pt
singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER --model /models/$MODEL --csv $WORKINGDIR/MNIST_stats/Model1-poison_ALL_best_particle_image_denoise.csv --outfile $WORKINGDIR/MNIST_stats/Model1-Poison_Model5-Performance.tsv

