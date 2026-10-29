from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch


def evaluate(model, data_loader, loss_fn, device):
    ## An evaluator used to monitor model's performence on a selected dataset with a given loss function.
    model.eval()
    correct_ans = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            flat_images = flatten(images)
            scores = model(flat_images)
            predictions = scores.argmax(dim=1)
            loss = loss_fn(scores, labels)
            loss_sum += loss.item() * labels.shape[0]
            correct_ans += (predictions == labels).sum().item()
            total += labels.shape[0]

    avg_loss = loss_sum / total
    accuracy = correct_ans / total
    return correct_ans, total, avg_loss, accuracy


def create_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

to_tensor = transforms.ToTensor()

flatten = nn.Flatten(start_dim=1)

model = create_model().to(device)

##linear = nn.Linear(in_features=28**2, out_features=10).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

full_train_data = datasets.MNIST(
    root="./data", train=True, download=False, transform=to_tensor
)

train_data, val_data = random_split(
    full_train_data, lengths=[50000, 10000], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)

epochs = 200
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(epochs):
    print("\nEpoch:", epoch + 1, "/", epochs)

    model.train()
    training_step = 0
    train_loss_sum = 0.0
    train_total = 0

    print("----------Training starts----------\n")

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        flat_images = flatten(images)
        scores = model(flat_images)
        predictions = scores.argmax(dim=1)
        loss = loss_fn(scores, labels)
        train_loss_sum += loss.item() * labels.shape[0]
        train_total += labels.shape[0]
        loss.backward()
        optimizer.step()

        training_step += 1
        # if training_step % 100 == 0:
        #    print("batch:", training_step, "loss:", loss.item())

    print("Training completed with " + str(training_step) + " steps.")
    print("Average training loss: " + str(train_loss_sum / train_total))

    print("----------Validation starts----------\n")

    (val_correct_ans, val_total, avg_val_loss, val_accuracy) = evaluate(
        model, val_loader, loss_fn, device
    )
    print("Cases tested: ", val_total)
    print("Cases passed: ", val_correct_ans)
    print("Accuracy: ", val_accuracy)
    print("Average loss: ", avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        state = model.state_dict()
        torch.save(state, "./mnist_mlp_training.pt")
        print(
            "Parameters in epoch",
            best_epoch,
            "is saved to ./mnist_mlp_training.pt",
        )

print("Training and validation done in", epochs, "epochs.\n")
print("Best epoch: ", best_epoch)
print("Best validation loss: ", best_val_loss)
##state = linear.state_dict()
##torch.save(state, "./mnist_linear_training.pt")
##print("Parameters saved to ./mnist_linear_training.pt")

print("\n----------Testing starts----------\n")

best_model = create_model().to(device)

saved_state = torch.load(
    "./mnist_mlp_training.pt", weights_only=True, map_location=device
)
best_model.load_state_dict(saved_state)
best_model.eval()

test_data = datasets.MNIST(
    root="./data", train=False, download=False, transform=to_tensor
)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

(test_correct_ans, test_total, avg_test_loss, test_accuracy) = evaluate(
    best_model, test_loader, loss_fn, device
)
print("Cases tested: ", test_total)
print("Cases passed: ", test_correct_ans)
print("Accuracy: ", test_accuracy)
print("Average loss: ", avg_test_loss)
