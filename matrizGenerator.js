const fs = require('fs');  // import the fs module

const n = 3;  // number of columns in the matrix
const matrix = [];  // initialize an empty matrix

let eixoY = [];

for (let i = 0; i <= 100; i++) {  // outer loop to iterate through rows
  for (let j = 0; j <= 100; j++) {  // inner loop to iterate through columns
    for (let k = 0; k <= 100; k++) {  // inner loop to iterate through columns
      // somar média ponderada dos 3 valores e preencher em uma nova coluna
      // pesos 3, 3, 4
      let acao = 0;
      let media_ponderada = (i * 1 + j * 1 + k * 8) / (1 + 1 + 8);

      if(media_ponderada < 65){ //ativa
        acao = 1; 
      }else{  // nao ativa
        acao = 0;
      }

      matrix.push([i, j, k, media_ponderada]);  // append the combination to the matrix
      eixoY.push(acao);
    }
  }
}

// create a string with the matrix data in CSV format
const csvData = matrix.map(row => row.join(', ')).join('\n');

// write the CSV data to a file
fs.writeFileSync('matrix.csv', csvData);

console.log('Matrix data has been written to matrix.csv');