<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Persian Handwriting Recognition</title>
</head>
<body>
    <h1>Persian Handwriting Recognition using CNN-BLSTM Hybrid Model</h1>
    <p>This repository contains the code and resources for a project aimed at recognizing Persian handwritten characters. The project employs a hybrid model integrating Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory (BLSTM) networks to capture both spatial and sequential features essential for accurate recognition of Persian script.</p>
    <h2>Overview</h2>
    <p>Persian handwriting recognition presents significant challenges due to the script's complexity, which includes 32 letters, each with four different positions (initial, middle, final, and isolated) and various dot placements. This project introduces an innovative approach to addressing these challenges by leveraging a hybrid CNN-BLSTM model.</p>
    <h2>Features</h2>
    <ul>
        <li><strong>Convolutional Neural Networks (CNNs):</strong> Used for local higher-level feature extraction from spatial input.</li>
        <li><strong>Bidirectional Long Short-Term Memory (BLSTM):</strong> Captures sequential patterns and dependencies in the data, considering both past and future context.</li>
        <li><strong>Data Augmentation:</strong> Utilizes <code>ImageDataGenerator</code> for real-time data augmentation, which includes rescaling, shearing, zooming, and horizontal flipping.</li>
    </ul>
    <h2>Model Architecture</h2>
    <p>The model architecture includes multiple convolutional layers for feature extraction, followed by MaxPooling layers to reduce spatial dimensions. A TimeDistributed layer is used to pass information to the BLSTM network, which captures sequential dependencies. Finally, Dense layers with activation functions are employed for classification.</p>

    <h2>Getting Started</h2>
    <h3>Prerequisites</h3>
    <ul>
        <li>Python 3.x</li>
        <li>TensorFlow</li>
        <li>Keras</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
        <li>Visualkeras</li>
    </ul>
    <h3>Installation</h3>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/yourusername/persian-handwriting-recognition.git
cd persian-handwriting-recognition
            </code></pre>
        </li>
        <li>Install the required packages:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
    </ol>
    <h3>Dataset</h3>
    <p>Ensure you have the dataset structured as follows:</p>
    <pre><code>d:/hudc2/
├── train/
|   ├── class1/
|   ├── class2/
|   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
    </code></pre>
    <h3>Training the Model</h3>
    <ol>
        <li>Set the paths to your dataset in the script:
            <pre><code>train_data_dir = 'd:/hudc2/train'
validation_data_dir = 'd:/hudc2/test'
            </code></pre>
        </li>
        <li>Run the training script:
            <pre><code>python train.py</code></pre>
        </li>
    </ol>
    <h3>Model Summary and Visualization</h3>
    <p>The model summary and architecture visualization are included in the script using <code>visualkeras</code>.</p>
    <h2>Evaluation</h2>
    <p>The model is evaluated using accuracy, precision, and recall metrics. The results are validated using the validation dataset.</p>
    <h2>Results</h2>
    <p>The model achieves a recognition accuracy of 98.8% on the combined dataset, demonstrating the effectiveness of using both natural and synthesized datasets in training robust handwriting recognition systems.</p>
    <h2>References</h2>
    <p>If you use this code in your research, please cite the following paper:</p>
    <blockquote>
        <p>Mohseni, Aida. "Optimizing Persian Handwriting Recognition: Leveraging a New Dataset with a CNN-BLSTM Hybrid Model."</p>
    </blockquote>
    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    <h2>Acknowledgments</h2>
    <p>Special thanks to the contributors and the open-source community for their support and tools.</p>
</body>
</html>
