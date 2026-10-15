from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch

torch.manual_seed(42)

to_tensor = transforms.ToTensor()

flatten = nn.Flatten(start_dim=1)

linear = nn.Linear(in_features=28**2, out_features=10)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

train_data = datasets.MNIST(
    root="./data", train=True, download=False, transform=to_tensor
)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

training_step = 0

print("----------Training starts----------\n")

for images, labels in train_loader:
    optimizer.zero_grad()
    flat_images = flatten(images)
    scores = linear(flat_images)
    predictions = scores.argmax(dim=1)
    loss = loss_fn(scores, labels)
    loss.backward()
    optimizer.step()

    training_step += 1
    if training_step % 100 == 0:
        print("batch:", training_step, "loss:", loss.item())

print("Training completed with " + str(training_step) + " steps.")

test_data = datasets.MNIST(
    root="./data", train=False, download=False, transform=to_tensor
)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

linear.eval()
correct_ans = 0
total = 0
print("\n----------Testing starts----------\n")
with torch.no_grad():
    for images, labels in test_loader:
        flat_images = flatten(images)
        scores = linear(flat_images)
        predictions = scores.argmax(dim=1)
        correct_ans += (predictions == labels).sum().item()
        total += labels.shape[0]

print("Cases tested: " + str(total))
print("Cases passed: " + str(correct_ans))
print("Accuracy: " + str(correct_ans / total) + "\n")

state = linear.state_dict()
torch.save(state, "/home/luming/project/pytorch exercise/mnist_linear_training.pt")
print(
    "Parameters saved to /home/luming/project/pytorch exercise/mnist_linear_training.pt"
)
