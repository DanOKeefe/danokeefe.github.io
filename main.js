// columns are: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, and Label
// file has no header.
const csvUrl =
'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv';

// Define the model globally
const model = tf.sequential();

async function LoadAndTrainData() {
  // We want to predict the column "Species"
  // It's read in as a string

  document.getElementById("training").innerHTML = '<p>Loading data...<p>'

  const csvDataset = tf.data.csv(
    csvUrl, {
      hasHeader: false,
      columnNames: ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'],
      columnConfigs: {
        Species: {
          isLabel: true
        }
      }
    });

    console.log('Data loaded!');

    // shuffle rows randomly
    //csvDataset = csvDataset.shuffle()

    // split into train/test
    //train_size = int(0.7 * DATASET_SIZE)

  // Number of features is the number of column names minus one for the label column
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
    .map(({xs, ys}) =>
      {
        // one-hot encode the target variable
        let y_out = []
        y = Object.values(ys)[0];
        if (y==='Iris-setosa') {
          y_out = [1,0,0];
        } else if (y==='Iris-versicolor') {
          y_out = [0,1,0];
        } else {
          y_out = [0,0,1];
        }

        // Convert xs(features) and ys(labels) from object form (keyed by
        // column name) to array form.
        return {xs:Object.values(xs), ys:y_out};
      })
    .batch(10);

  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 10,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax',
  }));

  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy'
  });

  console.log('Starting training:');

  // Fit model
  num_epochs = 32;
  return await model.fitDataset(flattenedDataset, {
    epochs: num_epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch}, Loss: ${logs.loss}`);
        document.getElementById("training")
          .innerHTML = `<p>Training... epoch ${epoch}/${num_epochs} Loss = ${logs.loss.toFixed(2)}</p>`;
      }
    }
  }), 
  doneTraining();
}

function doneTraining() {
  // Inform user that the model is done training
  document.getElementById('training').innerHTML = `<p>Done training! Ready for predictions.<p>`;
  // Insert the form for user input
  document.getElementById('inputForm').innerHTML =`
    <form>
      <label for="SepalLengthCm">Sepal Length (cm):</label>
      <input type="number" id="SepalLengthCm" name="SepalLengthCm" value=5 step="0.1" min=0 max=10>
      <br>
      <label for="SepalWidthCm">Sepal Width (cm):</label>
      <input type="number" id="SepalWidthCm" name="SepalWidthCm" value=4 step="0.1" min=0 max=5>
      <br>
      <label for="PetalLengthCm">Petal Length (cm):</label>
      <input type="number" id="PetalLengthCm" name="PetalLengthCm" value=1.4 step="0.1" min=0 max=5>
      <br>
      <label for="PetalWidthCm">Petal Width (cm):</label>
      <input type="number" id="PetalWidthCm" name="PetalWidthCm" value = 0.3 step="0.1" min=0 max=5>
      <br>
      <input type="submit" value="Predict" onclick="makePred(event)">
      <p id="pred">Prediction: </p>
      <div id="prediction"></div>
    </form> 
  `;
}


// Define function to run inference

async function makePred(e)  {
  // Don't refresh page
  e.preventDefault()
  const SepalLengthCm = parseFloat(document.getElementById('SepalLengthCm').value);
  const SepalWidthCm = parseFloat(document.getElementById('SepalWidthCm').value);
  const PetalLengthCm = parseFloat(document.getElementById('PetalLengthCm').value);
  const PetalWidthCm = parseFloat(document.getElementById('PetalWidthCm').value);
  // console.log(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm);
  const input = tf.tensor1d([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]);
  const result = model.predict(input.reshape(([1,4])));
  // console.log(result.dataSync()[0])
  // console.log(result.dataSync()[1])
  // console.log(result.dataSync()[2])
  // The the index of the maximum value of the output tensor
  const maxIdx = tf.argMax(result, axis=1).dataSync();

  // Convert maxIdx integer to string of the corresponding flower
  let predFlower = 'None'
  if (maxIdx[0]===0) {
    predFlower = 'Iris-setosa';
  } else if (maxIdx[0]===1) {
    predFlower = 'Iris-versicolor';
  } else if (maxIdx[0]===2) {
    predFlower = 'Iris-virginica';
  }

  document.getElementById('prediction').innerHTML = `<p>${String(predFlower)}</p>`;

  // Plot data with Plotly
  var data = [
    {
      x: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
      y: [result.dataSync()[0], result.dataSync()[1], result.dataSync()[2]],
      type: 'bar'
    }
  ];

  config = {
    displaylogo: false,
    plot_bgcolor: '#ffedd1',
  };
  
  Plotly.newPlot(
    'confidencePlot', 
    data, 
    config
  );
}

