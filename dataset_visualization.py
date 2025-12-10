import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from pathlib import Path

def visualize_class_samples(dataset, num_samples=5, figsize=(15, 10)):
    """
    Visualize random samples from each class
    
    Args:
        dataset: DriverBehaviorDataset instance
        num_samples: Number of samples to show per class
        figsize: Figure size
    """
    num_classes = 10
    fig, axes = plt.subplots(num_classes, num_samples, figsize=figsize)
    fig.suptitle('Sample Images per Class (Driver Behavior)', fontsize=16, y=0.995)
    
    # Group samples by class
    class_samples = {i: [] for i in range(num_classes)}
    for idx, sample in enumerate(dataset.samples):
        class_samples[sample['label']].append(idx)
    
    # Plot samples
    for class_idx in range(num_classes):
        indices = np.random.choice(class_samples[class_idx], 
                                   min(num_samples, len(class_samples[class_idx])), 
                                   replace=False)
        
        for col_idx, sample_idx in enumerate(indices):
            sample_info = dataset.get_sample_info(sample_idx)
            img = Image.open(sample_info['path'])
            
            ax = axes[class_idx, col_idx] if num_classes > 1 else axes[col_idx]
            ax.imshow(img)
            ax.axis('off')
            
            if col_idx == 0:
                # Get descriptive class name
                class_name = dataset.get_class_name(class_idx)
                ax.set_ylabel(f"c{class_idx}\n{class_name}", 
                            fontsize=8, rotation=0, labelpad=60, ha='right')
            
            # Add person name on first row
            if class_idx == 0:
                ax.set_title(f"{sample_info['person']}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_person_samples(dataset_root, person_name, num_samples_per_class=2, figsize=(15, 8)):
    """
    Visualize samples from a specific person across all classes
    
    Args:
        dataset_root: Path to HowDir folder
        person_name: Name of the person to visualize
        num_samples_per_class: Number of samples to show per class
    """
    root = Path(dataset_root)
    person_dir = root / person_name
    
    num_classes = 10
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=figsize)
    fig.suptitle(f'Samples from Person: {person_name}', fontsize=16)
    
    for class_idx in range(num_classes):
        class_dir = person_dir / f'c{class_idx}'
        img_files = list(class_dir.glob('*.jpg'))[:num_samples_per_class]
        
        for col_idx, img_file in enumerate(img_files):
            img = Image.open(img_file)
            
            ax = axes[class_idx, col_idx] if num_samples_per_class > 1 else axes[class_idx]
            ax.imshow(img)
            ax.axis('off')
            
            if col_idx == 0:
                ax.set_ylabel(f'c{class_idx}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'person_{person_name}_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_batch(dataloader, num_batches=1):
    """
    Visualize batches from dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to visualize
    """
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        batch_size = images.shape[0]
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Batch {batch_idx + 1}', fontsize=16)
        
        for idx in range(batch_size):
            row = idx // grid_size
            col = idx % grid_size
            
            # Denormalize image
            img = images[idx].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax = axes[row, col] if grid_size > 1 else axes
            ax.imshow(img)
            ax.set_title(f'Class: {labels[idx].item()}', fontsize=9)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(batch_size, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()


def plot_class_distribution(train_dataset, val_dataset, test_dataset):
    """
    Plot class distribution across splits
    """
    def get_class_counts(dataset):
        counts = np.zeros(10)
        for sample in dataset.samples:
            counts[sample['label']] += 1
        return counts
    
    train_counts = get_class_counts(train_dataset)
    val_counts = get_class_counts(val_dataset)
    test_counts = get_class_counts(test_dataset)
    
    x = np.arange(10)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax.bar(x, val_counts, width, label='Val', alpha=0.8)
    ax.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution Across Splits', fontsize=14)
    ax.set_xticks(x)
    
    # Use descriptive class names
    class_labels = [f"c{i}\n{train_dataset.CLASS_NAMES[i]}" for i in range(10)]
    ax.set_xticklabels(class_labels, fontsize=8, rotation=45, ha='right')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    from howdriv_viz import DriverBehaviorDataset, create_person_splits
    from torchvision import transforms
    
    dataset_root = '/kaggle/input/howdir-safedriving-benchmarking-sample/HowDir'
    all_persons = ['zakaria', 'hajar', 'sara', 'hamza', 'achraf', 
                   'mohammed', 'amal', 'youssef', 'mohssine']
    
    # Create splits
    train_persons, val_persons, test_persons = create_person_splits(all_persons)
    
    # Create dataset without normalization for visualization
    transform_viz = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_dataset = DriverBehaviorDataset(dataset_root, train_persons, transform_viz)
    
    # Visualize
    print("Visualizing class samples...")
    visualize_class_samples(train_dataset, num_samples=5)
    
    print("Visualizing specific person...")
    visualize_person_samples(dataset_root, 'zakaria', num_samples_per_class=3)
    
    print("Visualizing class distribution...")
    val_dataset = DriverBehaviorDataset(dataset_root, val_persons, transform_viz)
    test_dataset = DriverBehaviorDataset(dataset_root, test_persons, transform_viz)
    plot_class_distribution(train_dataset, val_dataset, test_dataset)