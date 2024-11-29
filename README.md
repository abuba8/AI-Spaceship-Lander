# AI Space Lander 🚀

This project implements an AI model to control a space lander in a simulation game. The model was developed from scratch, using a neural network that predicts the best moves based on game state inputs. The project involved collecting a dataset of over 200,000 rows, performing grid search over 72 parameter combinations, and integrating the trained model into the game to play autonomously.

## Demo
[Watch the AI Space Lander Gameplay Video](Recording.mp4)

## Features
- **Custom Neural Network**: Developed a multilayer perceptron (MLP) from scratch.
- **Grid Search Optimization**: Tuned hyperparameters (learning rate, momentum, hidden neurons) to find the best-performing model.
- **Integration in Gameplay**: The trained model is used to control the space lander in real-time.
- **Extensive Dataset**: Created and used a dataset with over 200,000 samples for training and evaluation.
- **Video Demonstration**: [Watch the AI in action on GitHub](#).

---

## Dataset

The dataset consists of game states and corresponding control actions:

- **Game States**: Features such as x, y, displacement x, and displacement y.
- **Actions**: Outputs indicating displacement x, and displacement y.
  
---

## Training the Model

The training process involves:

1. Initializing the neural network architecture with layers, activations, and loss functions.
2. Running grid search across combinations of:
   - Learning rate
   - Momentum
   - Number of hidden neurons
3. Training for 20 epochs per combination.
4. Selecting the best model based on validation loss.

### Neural Network Highlights

- **Forward and Backward Propagation**: Implemented manually for a transparent and customizable training process.
- **Stopping Criterion**: Training stops early if validation loss stagnates.

To train the model, run the trainer.ipynb file.
