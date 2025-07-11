import argparse
import os
from train import load_data,  train_model, evaluate_model
from taint import adversarial_attack_blackbox
from models import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3], help='Experiment number (1, 2, or 3)')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--particles', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    # get the experiment model 1 is MNIST, 2 is MNIST with data augmentation, 3 is AudioMNIST
    args.model_experiment = args.experiment
    args.save_dir = os.path.join(args.save_dir, f'experiment_{args.model_experiment}')
    if args.model_experiment == 1:
        args.model_path = args.model_path or 'mnist_model_1.keras'
        model = load_MNIST_model(args.model_path)
        train_ds, test_ds = load_data(experiment=1)
    elif args.model_experiment == 2:
        args.model_path = args.model_path or 'mnist_model_2.keras'
        model = load_MNIST_model(args.model_path)
        train_ds, test_ds = load_data(experiment=2)
    elif args.model_experiment == 3:
        args.model_path = args.model_path or 'mnist_model_3.keras'
        model = load_AudioMNIST_model(args.model_path)
        train_ds, test_ds = load_data(experiment=3)
    else:
        raise ValueError("Invalid experiment number. Choose 1, 2, or 3.")

    os.makedirs(args.save_dir, exist_ok=True)
    # if model path exists, load the model else train the model
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_weights(args.model_path)
    else:
        print("Training model...")
        model = train_model(model, train_ds, epochs=10)
        model.save(args.model_path)
    
    # Evaluate the model
    print("Evaluating model...") 
    evaluate_model(model, test_ds)

    adversarial_attack_blackbox(model, test_ds, image_index=0, output_dir=args.save_dir, num_iterations=args.iterations, num_particles=args.particles)

if __name__ == '__main__':
    main()
