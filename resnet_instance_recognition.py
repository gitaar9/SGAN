import argparse
import copy
import glob
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torchvision import models, transforms

from resnet_instance_classification.recognition_dataset import ShapeNetCarsRecognitionDataset, \
    ShapeNetCarsRecognitionDatasetOnlyLastImageIsTest, ShapeNetCarsGeneratedValidationSet, \
    ShapeNetCarsRecognitionDatasetOnlyLastImageIsTestWithExtra
from resnet_instance_classification.simple_dataset import SimpleShapenetRecognitionDataset


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False, verbose=True):
    use_extra_val = False
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            extra_running_corrects = 0

            # Iterate over data.
            if use_extra_val and phase == 'val':
                iterator = zip(dataloaders[phase], dataloaders['extra_val'])
            else:
                iterator = zip(dataloaders[phase], [[None, None]] * len(dataloaders[phase]))

            for (inputs, labels), (extra_inputs, _) in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if extra_inputs is not None:
                    extra_inputs = extra_inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    extra_preds = None
                    if use_extra_val and phase == 'val':
                        extra_outputs = model(extra_inputs)
                        # _, extra_preds = torch.max(extra_outputs, 1)
                        _, extra_preds = torch.max(outputs + extra_outputs * .5, 1)
                        # print(f"{max(preds)}:{max(extra_preds)}")

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if extra_preds is not None:
                    extra_running_corrects += torch.sum(extra_preds == labels.data)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if extra_running_corrects != 0:
                epoch_extra_acc = extra_running_corrects.double() / len(dataloaders['extra_val'].dataset)
            else:
                epoch_extra_acc = 0

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f} ({:.4f})'.format(phase, epoch_loss, epoch_acc, epoch_extra_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        if verbose:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc


def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = not feature_extracting


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        # model_ft = models.resnet34(pretrained=use_pretrained)
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def load_data(root_dir, extra_root_dir=None, amount_train_images=30, amount_extra_train_images=0,
              amount_validation_images=1, start_validation_idx=60, input_size=224, batch_size=8):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'extra_val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        'train': SimpleShapenetRecognitionDataset(
            root_dir,
            extra_root_dir=extra_root_dir,
            start_idx=0,
            amount_images=amount_train_images,
            amount_extra_images=amount_extra_train_images,
            transform=data_transforms['train']
        ),
        'val': SimpleShapenetRecognitionDataset(
            root_dir,
            extra_root_dir=None,
            start_idx=start_validation_idx,
            amount_images=amount_validation_images,
            amount_extra_images=0,
            transform=data_transforms['val']
        ),
    }

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=x == 'train', num_workers=4)
        for x in ['train', 'val']
    }


    # start_idx = 25
    # idxs = range(start_idx, start_idx+5)
    # for idx in idxs:
    #     image_datasets['train'].show_image(idx * amount_train_images)
    #     # if amount_extra_train_images > 0:
    #     #     image_datasets['train'].show_image(len(image_datasets['train']) // 2 + idx * amount_train_images)
    #     #     print(len(image_datasets['val']))
    #     image_datasets['val'].show_image(idx * amount_validation_images)

    return dataloaders_dict


def set_feature_extract_model(feature_extract, model):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                pass
                # print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    return optimizer_ft


def main(root_dir, extra_root_dir=None, amount_train_images=30, amount_extra_train_images=0, amount_validation_images=1,
         start_validation_idx=60, batch_size=8, num_epochs=15, initial_feature_extract_epochs=3, verbose=True):
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    # Number of classes in the dataset
    num_classes = len(glob.glob(os.path.join(root_dir, '*')))
    print(f"Found {num_classes} different classes")

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # create dataloaders
    dataloaders_dict = load_data(
        root_dir=root_dir,
        extra_root_dir=extra_root_dir,
        amount_train_images=amount_train_images,
        amount_extra_train_images=amount_extra_train_images,
        amount_validation_images=amount_validation_images,
        start_validation_idx=start_validation_idx,
        input_size=input_size,
        batch_size=batch_size
    )

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # TRAIN ONLY LAST LAYER
    # Set stg optimizer for either all or only last layer
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    optimizer_ft = set_feature_extract_model(feature_extract=True, model=model_ft)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, _ = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        device=device,
        num_epochs=initial_feature_extract_epochs,
        is_inception=(model_name == "inception"),
        verbose=verbose
    )

    # TRAIN ALL LAYERS
    set_parameter_requires_grad(model_ft, feature_extracting=False)
    optimizer_ft = set_feature_extract_model(feature_extract=False, model=model_ft)
    # Train and evaluate
    model_ft, hist, best_accuracy = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        device=device,
        num_epochs=num_epochs - initial_feature_extract_epochs,
        is_inception=(model_name == "inception"),
        verbose=verbose
    )

    return best_accuracy.cpu()


if __name__ == '__main__':
    # root_dir, extra_root_dir=None, amount_train_images=30, amount_extra_train_images=0, amount_validation_images=1, start_validation_idx=60
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--extra_root_dir', type=str, default=None)

    parser.add_argument('--amount_train_images', type=int, default=30)
    parser.add_argument('--amount_extra_train_images', type=int, default=0)
    parser.add_argument('--amount_validation_images', type=int, default=1)
    parser.add_argument('--start_validation_idx', type=int, default=60)

    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training (change depending on how much memory you have)")
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of epochs to train for")
    parser.add_argument('--initial_feature_extract_epochs', type=int, default=2,
                        help="Number of epochs to train only last layer")

    parser.add_argument('--num_runs', type=int, default=1,
                        help="How many times to repeat the experiment")

    opt = parser.parse_args()

    final_best_accs = []
    for run in range(opt.num_runs):
        print(f"{run + 1}/{opt.num_runs}")
        best_acc = main(
            root_dir=opt.root_dir,
            extra_root_dir=opt.extra_root_dir,
            amount_train_images=opt.amount_train_images,
            amount_extra_train_images=opt.amount_extra_train_images,
            amount_validation_images=opt.amount_validation_images,
            start_validation_idx=opt.start_validation_idx,
            batch_size=opt.batch_size,
            num_epochs=opt.num_epochs,
            initial_feature_extract_epochs=opt.initial_feature_extract_epochs,
            verbose=opt.num_runs == 1
        )
        final_best_accs.append(best_acc)

    print('#'*20)
    final_best_accs = np.asarray(final_best_accs)
    print(f"FINAL BEST ACCURACY: {np.mean(final_best_accs)} +- {np.std(final_best_accs)}")
    print(f"({np.mean(final_best_accs) * 100:.2f} $\\pm$ {np.std(final_best_accs) * 100:.2f})")

# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_test_set/cars_test --num_epochs=10
# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_test_set/cars_test --num_epochs=10 --amount_train_images=60
# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_test_set/cars_test --num_epochs=10 --amount_extra_train_images=30 --extra_root_dir=/samsung_hdd/Files/AI/TNO/remote_folders/pixel_nerf_eval/pixel_nerf_eval_output/sncar/data_augmentation

# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_drone_test_set/cars_test --num_epochs=15 --num_runs=4 --amount_validation_images=10
# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_drone_test_set/cars_test --num_epochs=15 --num_runs=4 --amount_validation_images=10 --amount_extra_train_images=30 --extra_root_dir=/samsung_hdd/Files/AI/TNO/remote_folders/pixel_nerf_eval/pixel_nerf_eval_output/sncar/drone/data_augmentation_sym_loss
# python resnet_instance_recognition.py --root_dir=/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_recognition_drone_test_set/cars_test --num_epochs=15 --num_runs=4 --amount_validation_images=10 --amount_extra_train_images=30 --extra_root_dir=/samsung_hdd/Files/AI/TNO/remote_folders/pixel_nerf_eval/pixel_nerf_eval_output/sncar/drone/data_augmentation_no_mirror

