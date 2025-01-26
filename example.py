import torch
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset
from Adversarial_Observation.utils import load_MNIST_model, load_data
from Adversarial_Observation import AdversarialTester, ParticleSwarm


def adversarial_attack_whitebox(model: torch.nn.Module, dataloader: DataLoader) -> None:
    """
    Performs a white-box adversarial attack on the model using AdversarialTester.
    
    Args:
        model (torch.nn.Module): The trained model to attack.
        dataloader (DataLoader): The data loader containing the dataset.
    """
    # Initialize the AdversarialTester with the model
    attacker = AdversarialTester(model)

    # Perform the attack on the dataset
    for images, _ in dataloader:
        attacker.test_attack(images)


def adversarial_attack_blackbox(model: torch.nn.Module, dataloader: DataLoader) -> DataLoader:
    """
    Performs a black-box adversarial attack on the model using Particle Swarm optimization.
    
    Args:
        model (torch.nn.Module): The trained model to attack.
        dataloader (DataLoader): The data loader containing the dataset.
    
    Returns:
        DataLoader: A dataloader containing adversarially perturbed images.
    """
    # Get the first two images from the dataset to simulate misclassification
    single_image_input = dataloader.dataset[0][0]
    single_image_target = torch.argmax(model(single_image_input.unsqueeze(0)))

    single_misclassification_input = dataloader.dataset[1][0]
    single_misclassification_target = torch.argmax(model(single_misclassification_input.unsqueeze(0)))

    # Ensure the targets are different to simulate misclassification
    assert single_image_target != single_misclassification_target, \
        "Target classes should be different for misclassification."

    # Create a noisy input set for black-box attack
    input_set = [single_image_input + torch.randn_like(single_image_input) for _ in range(100)]
    input_set = torch.stack(input_set)

    print(f"Target class for original image: {single_image_target}")
    print(f"Target class for misclassified image: {single_misclassification_target}")
    
    # Initialize the Particle Swarm optimizer with the model and input set
    attacker = ParticleSwarm(
        model, input_set, single_misclassification_target, num_iterations=30,
        epsilon=0.8, save_dir='results', inertia_weight=0.8, cognitive_weight=0.5,
        social_weight=0.5, momentum=0.9, velocity_clamp=0.1
    )
    attacker.optimize()

    # Generate adversarial dataset
    return get_adversarial_dataloader(attacker, model, single_misclassification_target, single_image_target)


def get_adversarial_dataloader(attacker: ParticleSwarm, model: torch.nn.Module, target_class: int, original_class: int) -> DataLoader:
    """
    Generates a DataLoader containing adversarially perturbed images.
    
    Args:
        attacker (ParticleSwarm): The ParticleSwarm instance after optimization.
        model (torch.nn.Module): The trained model used for evaluating adversarial examples.
        target_class (int): The target class for the attack.
        original_class (int): The original class of the image.
    
    Returns:
        DataLoader: A dataset containing adversarial images with their target and original class confidences.
    """
    print(f"Generating adversarial examples with target class {target_class} and original class {original_class}")

    images, target_confidence, original_confidence = [], [], []

    for particle in attacker.particles:
        for position in particle.history:
            output = model(position)
            if torch.argmax(output) == target_class:
                images.append(position)
                target_confidence.append(torch.softmax(output, dim=1)[target_class])
                original_confidence.append(torch.softmax(model(particle.original_data))[original_class])

    # Convert lists to tensors and return a TensorDataset
    X_images = torch.stack(images)
    X_original_confidence = torch.stack(original_confidence)
    y = torch.stack(target_confidence)

    return DataLoader(TensorDataset(X_images, y, X_original_confidence))


def train(model: torch.nn.Module, dataloader: DataLoader, epochs: int = 10) -> torch.nn.Module:
    """
    Trains the model for a specified number of epochs.
    
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): The data loader for the training data.
        epochs (int, optional): Number of training epochs. Defaults to 10.
    
    Returns:
        torch.nn.Module: The trained model.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        start_time = time.time()  # Track time for each epoch
        print(f"\nEpoch {epoch+1}/{epochs}:")
        
        running_loss = 0.0
        accuracy = 0

        # Use tqdm for a progress bar
        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for images, labels in pbar:
                optimizer.zero_grad()
                
                # Forward pass
                output = model(images)
                
                # Compute loss
                loss_val = loss_fn(output, labels)
                
                # Backward pass and optimization
                loss_val.backward()
                optimizer.step()

                running_loss += loss_val.item()
                accuracy += (output.argmax(dim=1) == labels).float().mean().item()

                # Update progress bar description
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=accuracy / (pbar.n + 1))
        
        # Print average loss and accuracy for the epoch
        epoch_loss = running_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed_time:.2f}s, Average Loss: {epoch_loss:.4f}, Accuracy: {accuracy / len(dataloader):.4f}")
    
    return model


def main() -> None:
    """
    Main function to execute the adversarial attack workflow.
    """
    # Load pre-trained model (MNIST model)
    model = load_MNIST_model()

    # Load MNIST dataset (train and test loaders)
    train_loader, test_loader = load_data()

    # Train the model
    model = train(model, train_loader, epochs=3)

    # Perform black-box attack using Particle Swarm optimization
    print("Performing black-box adversarial attack...")
    final_dataloader = adversarial_attack_blackbox(model, test_loader)
    


if __name__ == "__main__":
    main()
