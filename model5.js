const { rand } = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');



const fs = require('fs');
var colors = require('colors');

// Define the model architecture
let model = tf.sequential();
model.add(tf.layers.dense({units: 32, inputShape: [3]}));
model.add(tf.layers.dense({units: 10, activation: 'relu' }));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

async function trainModel(moistureData, labels) {
  // Train the model using the moisture data and labels
  const modelFolder = './irrigation-model-1';
  if(fs.existsSync(modelFolder)) {
    console.log(`+ Pasta do modelo "${modelFolder}" existe, carregando modelo...`.green.bold);
    model = await tf.loadLayersModel('file://irrigation-model-1/model.json');

  } else {
    console.log(`+ Pasta do modelo "${modelFolder}" não existe`.yellow.bold);
    const history = await model.fit(moistureData, tf.tensor(labels), {
      epochs: 5,
      validationSplit: 0.1,
      callbacks: {
        onTrainBegin: () => console.log('+ Treino iniciado'.green.bold),
        onTrainEnd: () => {
          console.log('===================================='.white.bold)
          console.log('+ Treino do modelo finalizado !'.white.bold)
        },
      }
    });

    console.log('+ Precisão da previsão: '.white.bold,(history.history.acc[history.history.acc.length - 1] * 100).toFixed(2).green.bold +" %".green.bold);
    console.log('===================================='.white.bold)
    const saveResults = await model.save('file://irrigation-model-1');
    console.log("+ Modelo salvo !".green.bold);
  }
 
}

async function predict(moistureData) {
  // Use the model to predict whether to activate the irrigation pump
  const prediction = model.predict(moistureData);
  if(prediction.dataSync()[0] > 0.8){
    console.log('+ Previsão :'.white+' Ativar'.green.bold, );
  }else{
    console.log('+ Previsão :'.white+' Não ativar'.red.bold, );
  }
  
}

// matriz considera 3 dias de analise do solo
const matrix = []; 
let eixoY = [];

for (let i = 0; i <= 100; i++) {  // outer loop to iterate through rows
  for (let j = 0; j <= 100; j++) {  // inner loop to iterate through columns
    for (let k = 0; k <= 100; k++) {  // inner loop to iterate through columns
      //@NOTE: equação que determina a ação de irrigação do modelo ( ultimo dia é o que mais pesa na consideração)
      // somar média ponderada dos 3 valores e preencher o eixo y com a acao
      // pesos 1 1 8
      let media_ponderada = (i * 1 + j * 1 + k * 8) / (1 + 1 + 8); 
      //@NOTE: valor "65" foi determinado perante analise de dados do arquivo matrix.csv
      // Esse valor será ajustado conforme testes práticos do algoritmo
      // valor 65 da media ponderada parece seguir um padrão ideal de variação de umidade do solo. abaixo disso é ideal efetuar a rega
      let acao = 0;
      if(media_ponderada < 65){ //ativa
        acao = 1;
      }else{// nao ativa
        acao = 0;
      }
      matrix.push([i, j, k]);// append the combination to the matrix
      eixoY.push(acao);
    }
  }
}

// Sample training data
const moistureData = tf.tensor2d(matrix);
const labels = eixoY;


// // Sample training data
// const moistureData = tf.tensor2d([
//     [80, 90, 90],
//     [100, 90, 90],
//     [100, 100, 90],
//   [75, 45, 55],
//   [80, 50, 60],
//   [85, 55, 65],
//   [90, 60, 70],
//   [95, 65, 75],
// ]);

// const labels = [
//     0,
//     0,
//     0,
//     1,
//     1, 
//     1, 
//     0, 
//     0, 
// ];


// Train the model
trainModel(moistureData, labels).then(async () => {

  function getRandomInt(max) {
      return Math.floor(Math.random() * max);
  }

  for (let i=1; i<=10; i++)  {
    let a = 30 + getRandomInt(20);
    let b = 30 + getRandomInt(30)
    let c = 33 + getRandomInt(40)
    
    console.log('Dado teste:  '+ a +" - "+ b +" - "+ c +''.yellow );
    let testData = tf.tensor2d([[a, b, c]]);
    predict(testData);
  }

console.log('===================================='.yellow);
console.log('+ Teste manual'.yellow);
console.log('Dado teste:  [80, 50, 40]'.yellow );
let testData2 = tf.tensor2d([[80, 50, 40]]);
predict(testData2);

console.log('Dado teste:  [80, 30, 30]'.yellow );
let testData3 = tf.tensor2d([[80, 30, 30]]);
predict(testData3);

console.log('Dado teste:  [40, 50, 40]'.yellow );
let testData4 = tf.tensor2d([[40, 50, 40]]);
predict(testData4);

console.log('Dado teste:  [80, 60, 80]'.yellow );
let testData5 = tf.tensor2d([[80, 60, 80]]);
predict(testData5);

console.log('Dado teste:  [80, 90, 100]'.yellow );
let testData6 = tf.tensor2d([[80, 90, 100]]);
predict(testData6);

console.log('Dado teste:  [10, 75, 65]'.yellow );
let testData7 = tf.tensor2d([[10, 75, 65]]);
predict(testData7);

console.log('Dado teste:  [70 - 60 - 65]'.yellow );
let testData8 = tf.tensor2d([[45, 40, 72]]);
predict(testData8);


});