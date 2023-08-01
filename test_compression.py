import time
import torch
from utils import get_network, get_test_dataloader
from conf import settings
import argparse

def calculate_average_inference_time_and_memory(net, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    num_samples = len(data_loader.dataset)
    total_inference_time = 0.0
    total_memory_consumption = 0.0

    # Measure inference time and memory consumption for each batch
    for batch_index, (images, labels) in enumerate(data_loader):
        images = images.to(device)  # Move the input tensor to the same device as the model

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = net(images)
        end_time = time.time()
        inference_time = end_time - start_time

        # Measure memory consumption
        memory_consumption = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB

        total_inference_time += inference_time
        total_memory_consumption += memory_consumption

    # Calculate average inference time and memory consumption per image
    average_inference_time = total_inference_time / num_samples * 1000  # Convert to ms
    average_memory_consumption = total_memory_consumption / num_samples

    return average_inference_time, average_memory_consumption

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    # Load the pruned model
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create an instance of the VGG model (same as in load_pretrained_model)
    model = get_network(args)

    # Load the pruned model
    model.load_state_dict(torch.load(args.weights, map_location=device))

    print(model)
    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # Compute top 5
            correct_5 += correct[:, :5].sum()

            # Compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    # Calculate average inference time and memory consumption per image
    average_inference_time, average_memory_consumption = calculate_average_inference_time_and_memory(model, cifar100_test_loader)

    # Print the results
    print(f"Average Inference Time per Image: {average_inference_time:.2f} ms")
    print(f"Average Memory Consumption per Image: {average_memory_consumption:.2f} MB")
