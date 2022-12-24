const tf = require('@tensorflow/tfjs');

async function makeDecision(humidity1, humidity2, humidity3) {
  // Normalize the input humidity levels between 0 and 1
  const inputs = tf.tensor2d([[humidity1 / 100, humidity2 / 100, humidity3 / 100]]);

  // Define the model architecture
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, inputShape: [3] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  // Compile the model
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  // Train the model
  const trainingData = tf.tensor2d([
    [1, 0, 0],   // humidities >= 70, < 50, < 60 -> activate water bomb pump
    [1, 1, 0],   // humidities >= 70, >= 50, < 60 -> do not activate water bomb pump
    [1, 1, 1],   // humidities >= 70, >= 50, >= 60 -> do not activate water bomb pump
    [0, 0, 0],   // humidities < 70, < 50, < 60 -> do not activate water bomb pump
    [0, 1, 0],   // humidities < 70, >= 50, < 60 -> activate water bomb pump
    [0, 1, 1],   // humidities < 70, >= 50, >= 60 -> do not activate water bomb pump
    [0, 0, 1],   // humidities < 70, < 50, >= 60 -> do not activate water bomb pump
  ]);
  const labels = tf.tensor1d([1, 0, 0, 0, 1, 0, 0]);
  await model.fit(trainingData, labels, { epochs: 10 });

  // Make a prediction
  const prediction = model.predict(inputs);
  const decision = prediction.dataSync()[0];

  if (decision > 0.5) {
    console.log('Activate water bomb pump');
  } else {
    console.log('Do not activate water bomb pump');
  }
}



makeDecision(70, 50, 60);  // this should print "Do not activate water bomb pump"