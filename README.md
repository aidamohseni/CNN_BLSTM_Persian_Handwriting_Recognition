# neuralnetworks
Neural Networks Python source code
This Code is written for Persian handwriting recognition 
The used dataset is Aira, which is available at https://drive.google.com/drive/folders/1QJ8nBMOfNVwnSrfvHFNsm_uLtyQVouSW?usp=drive_link
the total number of parameters in your CNN model for the provided layers is:
Total Parameters = Parameters in First Conv2D Layer + Parameters in Second Conv2D Layer
Total Parameters = 320 parameters + 18,496 parameters
Total Parameters = 18,816 parameters

e the output shape with the corrected configuration:

Bidirectional LSTM layer:

LSTM units: 64
Bidirectional: It processes the input sequence in both directions (forward and backward), effectively doubling the number of units to 128.
The input shape depends on the output shape of the previous layer, which in your case is (3, 128).
The output shape after the Bidirectional LSTM layer would be (3, 128), as the LSTM units combine their forward and backward outputs.

Dropout layer:

Dropout rate: 0.25
It doesn't change the shape of the data, so the output shape remains (3, 128).
Dense layer:

Number of units: 1
The output shape becomes (3, 1).
Activation layer (sigmoid):

Applies the sigmoid activation function to the output of the previous Dense layer.
The output shape remains (3, 1).

