const tf = require('@tensorflow/tfjs');

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({ units: 32, inputShape: [3] }));
// Add a hidden layer with 10 units
model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
// Add an output layer with 1 unit (for our binary prediction)
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model
model.compile({
  optimizer: 'Adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

async function trainModel(moistureData, labels ) {
    
    const history = await model.fit(tf.tensor(moistureData), tf.tensor(labels), {
        epochs: 100,
        validationSplit: 0.4,
        callbacks: {
          onTrainBegin: () => console.log('Training started'),
          onTrainEnd: () => console.log('Training completed'),
          onEpochEnd: (epoch, log) => {
            console.log(`
                Log:${Object.keys(log)}
                EPOCH (${epoch + 1}):
                Train Accuracy: ${(log.acc * 100).toFixed(2)},
                Val Accuracy:  ${(log.val_acc * 100).toFixed(2)},
                Val Loss = ${(log.val_loss * 100).toFixed(2)},
                Loss = ${(log.loss * 100).toFixed(2)}
              `);
          }
        }
      });

//   console.log('Precisão final: ', history.history.acc[history.history.acc.length - 1]);


  return history.history.acc[history.history.acc.length - 1];
}

async function predict(moistureData) {
  // Use the model to predict whether to activate the irrigation pump
  const prediction = model.predict(moistureData).round();
  const activation = prediction.dataSync()[0] > 0.5 ? 1 : 0;
  console.log(`Ativar bomba água: ${activation}`);

}

const trainingData = [
    [80, 70, 65], 
    [75, 80, 70],
    [60, 55, 50], 
    [90, 85, 80], 
    [70, 75, 80],
    [50, 40, 35],
    [80, 70, 70],
    [70, 80, 75],
    [90, 80, 80],
    [90, 80, 30],
    [40, 40, 30]
  ];
  
  // Define the labels for the training data
  const labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1];

trainModel(trainingData, labels).then((acurracy) => {
    console.log('Precisão Final: ' + acurracy);

    console.log('teste 1');
    const testData1 = tf.tensor2d([[80, 40, 80]]);
    predict(testData1);

    console.log('teste 2');
    const testData2 = tf.tensor2d([[60, 50, 40]]);
    predict(testData2);

    console.log('teste 3');
    const testData3 = tf.tensor2d([[60, 40, 30]]);
    predict(testData3);

    console.log('teste 4');
    const testData4 = tf.tensor2d([[80, 75, 70]]);
    predict(testData4);


    console.log('teste 5');
    const testData5 = tf.tensor2d([[80, 60, 40]]);
    predict(testData5);

  

});