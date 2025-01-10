from Adversarial_Observation.utils import load_pretrained_model, load_data, fgsm_attack, pgd_attack  # Assuming utils.py contains this function
from Adversarial_Observation import AdversarialTester
from Adversarial_Observation import ParticleSwarm
import torch

def adversarial_attack_whitebox(model, dataloader):
    # Initialize the AdversarialTester with the model
    attacker = AdversarialTester(model)

    # Perform the attack on the dataset
    for images, _ in dataloader:
        attacker.test_attack(images)

# Example function call
def adversarial_attack_blackbox(model, dataloader):
    single_image_input = dataloader.dataset[0][0]  # Get the first image from the dataset
    single_image_target = dataloader.dataset[0][1]  # Get the target label for the first image

    misclassiciation_target = 4

    input_set = [single_image_input + (torch.randn_like(single_image_input) * .001) for _ in range(10)]  # Create a set of 10 noisy images
    # convert input_set to a tensor
    input_set = torch.stack(input_set)
    
    # Initialize the Particle Swarm optimizer with the model and the input set
    attacker = ParticleSwarm(model, input_set, target_class=misclassiciation_target)
    final_perturbed_images = attacker.optimize()
    import pdb; pdb.set_trace()
    return final_perturbed_images
 

def main():
    # Load pre-trained model (ResNet18)
    model = load_pretrained_model()

    # Load CIFAR-10 validation data (using the transformed dataset)
    dataloader = load_data(batch_size=32)

    # Perform white-box attack using AdversarialTester
    # print("Performing white-box adversarial attack...")
    # adversarial_attack_whitebox(model, dataloader)

    # Perform black-box attack using Swarm
    print("Performing black-box adversarial attack...")
    adversarial_attack_blackbox(model, dataloader)

if __name__ == "__main__":
    main()
