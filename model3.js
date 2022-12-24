// Import the required libraries
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');


// Define the model

async function iniciar(){

    const model = tf.sequential();
    
    // Add an input layer with NUM_DAYS units
    model.add(tf.layers.dense({ units: 3, inputShape: [3] }));
    
    // Add a hidden layer with 10 units
    model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    
    // Add an output layer with 1 unit (for our binary prediction)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    // Compile the model using the binaryCrossentropy loss function and the sgd optimizer
    model.compile({ loss: 'binaryCrossentropy', optimizer: 'sgd' });

// Define the training data
const trainingData = [
  [80, 70, 65], // First humidity level is >= 80, so water pump should be activated
  [75, 80, 70], // First and second humidity levels are >= 70, so water pump should not be activated
  [60, 55, 50], // First humidity level is < 80, so water pump should not be activated
  [90, 85, 80], // First humidity level is >= 80, so water pump should be activated
  [70, 75, 80]  // First and second humidity levels are >= 70, so water pump should not be activated
];

// Define the labels for the training data
const labels = [1, 0, 1, 0, 0];

// Train the model
const history = await model.fit(tf.tensor(trainingData), tf.tensor(labels), {
  epochs: 10,
  validationSplit: 0.2,
  callbacks: {
    onTrainBegin: () => console.log('Training started'),
    onTrainEnd: () => console.log('Training completed'),
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});

// Get the accuracy of the model
const accuracy = _.last(history.history.acc);
console.log(`Accuracy: ${accuracy}`);

// Use the model to make a prediction
const input = tf.tensor2d([[80, 75, 70]]); // First humidity level is >= 80, so water pump should be activated
const prediction = model.predict(input);
const activation = prediction.dataSync()[0] > 0.5 ? 1 : 0;
console.log(`Activate water pump: ${activation}`);

// Visualize the accuracy in the terminal
console.log('Accuracy:');
console.log('=========');
console.log('teste');
console.log(history.history);

}

iniciar();