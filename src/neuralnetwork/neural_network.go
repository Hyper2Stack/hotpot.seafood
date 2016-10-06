package neuralnetwork

type NeuralNetwork interface {
   AddLayer (layer Layer) NeuralNetwork
   Fit (input, expect *SimpleMatrix, alpha float64) NeuralNetwork
   Predict (input *SimpleMatrix) *SimpleMatrix
   Learn (predict *SimpleMatrix, expect *SimpleMatrix) NeuralNetwork
   Update (alpha float64) NeuralNetwork
}
