const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');

async function iniciar(){
   
    const NUM_DAYS = 3; // Number of days to consider when making a prediction
    const IDEAL_HUMIDITY = 70; // Ideal soil humidity level
    
    // Next, we'll define our model using the TensorFlow.js Layers API
    const model = tf.sequential();
    
    // Add an input layer with NUM_DAYS units
    model.add(tf.layers.dense({ units: NUM_DAYS, inputShape: [NUM_DAYS] }));
    
    // Add a hidden layer with 10 units
    model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    
    // Add an output layer with 1 unit (for our binary prediction)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    // Compile the model using the binaryCrossentropy loss function and the sgd optimizer
    model.compile({ loss: 'binaryCrossentropy', optimizer: 'sgd' });
    
    // Now we'll define our training data. This should be an array of NUM_DAYS-element arrays,
    // where each element is the soil humidity level for a given day.
    const X_train = [  [80, 70, 60],
      [60, 70, 80],
      [70, 60, 70],
      [60, 60, 60],
      [70, 70, 70],
    ];
    
    // Our labels should be a binary array indicating whether or not we should activate the
    // water bomb pump for each corresponding set of soil humidity levels in X_train
    const y_train = [1, 0, 1, 0, 0];
    
    // Now we can train our model using the fit method
    const history = await model.fit(tf.tensor(X_train), tf.tensor(y_train), {
      epochs: 10,
      validationSplit: 0.2,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'val_loss'],
        { callbacks: ['onEpochEnd'] }
      ),
    });
    
    
    // Now that our model is trained, we can use it to make predictions on new data
    const X_test = [  [80, 70, 60],  // This should activate the water bomb pump
      [60, 70, 80],  // This should not activate the water bomb pump
      [70, 60, 70],  // This should activate the water bomb pump
    ];
    
    // Make predictions on the test data
    const y_pred = model.predict(tf.tensor(X_test)).round();
    
    // Print the predictions to the console
    console.log(y_pred.dataSync());  // Outputs: [1, 0, 1]
}

iniciar();
