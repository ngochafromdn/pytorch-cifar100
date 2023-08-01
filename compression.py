import torch
import torch.nn.utils.prune as prune
import argparse
from models.vgg import vgg16_bn

# Load the Pre-trained Model
def load_pretrained_model(model_path):
    model = vgg16_bn()  # Create an instance of the VGG model
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)

    model.eval()  # Set the model to evaluation mode
    return model

# Apply Model Compression Techniques

# Technique 1: Pruning
def apply_pruning(model, prune_percent=50):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percent / 100)

    # Do not remove pruning here. Removing pruning requires a pruned model to be fine-tuned first.

# Technique 2: Quantization
def apply_quantization(model):
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

# Technique 3: Weight Sharing
def apply_weight_sharing(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Share the weight of the convolutional layer with the next layer
            next_module = list(module.children())[0]
            if isinstance(next_module, torch.nn.Conv2d):
                next_module.weight = module.weight

# Technique 4: Add your custom compression technique here if needed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_path', type=str, required=True, help='path to the pretrained model')
    args = parser.parse_args()

    # Load the pre-trained model
    model = load_pretrained_model(args.pretrained_path)

    # Technique 1: Apply pruning with 50% weight pruning
    apply_pruning(model)

    # Technique 2: Apply quantization
    apply_quantization(model)

    # Technique 3: Apply weight sharing
    apply_weight_sharing(model)

    # Save the pruned, quantized, and weight-shared model
    torch.save(model.state_dict(), 'compressed_model.pth')
