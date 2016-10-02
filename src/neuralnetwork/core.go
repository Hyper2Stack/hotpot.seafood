package neuralnetwork

import "math"

// ref: https://github.com/andersbll/nnet

type Layer interface {
   // Setup layer with parameters that are unknown at 
   Setup ()
   // Output matrix shape M * N
   Dim () (int, int)
   // Calculate layer output for given input (forward propagation).
   ForwardProp (input *SimpleMatrix) *SimpleMatrix
   // Calculate input gradient.
   BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix
   // Update layer parameter gradients as calculated from BackwardProp(). 
   ParamsUpdate (learning_rate float64)
}

type LossMixin interface {
   // Calculate mean loss given output and predicted output.
   Loss (output, output_pred *SimpleMatrix) *SimpleMatrix
   // Calculate input gradient given output and predicted output.
   //InputGrad (output, output_pred *SimpleMatrix) *SimpleMatrix
}

type LayerLinear struct {
   Layer
   W, B *SimpleMatrix
   DeltaW, DeltaB *SimpleMatrix
   Last *SimpleMatrix
   WeightScale, WeightDecay float64
}

func NewLayerLinear (input_m, input_n, output_n int, weight_scale, weight_decay float64) *LayerLinear {
   // weight_decay default: 0.0
   c := new(LayerLinear)
   c.WeightScale = weight_scale
   c.WeightDecay = weight_decay
   c.W = NewSimpleMatrix(input_n, output_n).FillGuassian(0, weight_scale)
   c.B = NewSimpleMatrix(input_m, output_n)
   return c
}

func (c *LayerLinear) Setup () {
}

func (c *LayerLinear) Dim () (int, int) {
   return c.B.M, c.B.N
}

func (c *LayerLinear) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.Last = input
   return input.Dot(c.W).Add(c.B, 1, 1)
}

func (c *LayerLinear) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.DeltaW = c.Last.T().Dot(output_grad).Scale(1.0 / float64(output_grad.M)).Add(c.W.Scale(c.WeightDecay), 1, -1)
   c.DeltaB = output_grad.Clone()
   return output_grad.Dot(c.W.T())
}

func (c *LayerLinear) ParamsUpdate (learning_rate float64) {
   c.W = c.W.Add(c.DeltaW.Scale(learning_rate), 1, -1)
   c.B = c.B.Add(c.DeltaB.Scale(learning_rate), 1, -1)
}


type LayerActivation struct {
   Layer
   Fun, FunDerivative func (float64) float64
   Last *SimpleMatrix
}

func NewLayerActivation (input_m, input_n int, fun_type string) *LayerActivation {
   c := new(LayerActivation)
   c.Last = NewSimpleMatrix(input_m, input_n)
   switch fun_type {
   case "tanh":
      c.Fun = Tanh
      c.FunDerivative = TanhDerivative
   case "relu":
      c.Fun = Relu
      c.FunDerivative = ReluDerivative
   default:
      c.Fun = Sigmoid
      c.FunDerivative = SigmoidDerivative
   }
   return c
}

func (c *LayerActivation) Setup () {
}

func (c *LayerActivation) Dim () (int, int) {
   return c.Last.M, c.Last.N
}

func (c *LayerActivation) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.Last = input.Map(c.Fun)
   return c.Last.Clone()
}

func (c *LayerActivation) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   return output_grad.EltMul(c.Last.Map(c.FunDerivative))
}

func (c *LayerActivation) ParamsUpdate (learning_rate float64) {
}


type LayerLogRegression struct {
   Layer
   LossMixin
   M, N int
}

func NewLayerLogRegression (input_m, input_n int) *LayerLogRegression {
   c := new(LayerLogRegression)
   c.M = input_m
   c.N = input_n
   return c
}

func (c *LayerLogRegression) Setup () {
}

func (c *LayerLogRegression) Dim () (int, int) {
   return c.M, c.N
}

func (c *LayerLogRegression) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   return input.Softmax()
}

func (c *LayerLogRegression) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   // LogRegression does not support back-propagation of gradients.
   // It should occur only as the last layer of a NeuralNetwork.
   return nil
}

func __entropy_clip__ (x float64) float64 {
   if x < 1e-15 {
      return 1e-15
   }
   if x > 1 - 1e-15 {
      return 1 - 1e-15
   }
   return x
}
func (c *LayerLogRegression) Loss (output, output_pred *SimpleMatrix) *SimpleMatrix {
   m := output_pred.M
   loss := NewSimpleMatrix(m, 1)
   for i := output_pred.M - 1; i >= 0; i++ {
      row := output_pred.Row(i).Map(__entropy_clip__)
      loss.Data[i][0] = -row.Scale(1 / row.EltSum()).Map(math.Log).EltMul(output.Row(i)).EltSum()
      loss.Data[i][0] /= float64(m)
   }
   return loss
}

func (c *LayerLogRegression) ParamsUpdate (learning_rate float64) {
}

/* Layer Template
type LayerX struct {
   Layer
}

func NewLayerX (input_m, input_n, output_n int) *LayerX {
   c := new(LayerX)
   return c
}

func (c *LayerX) Setup () {
}

func (c *LayerX) Dim () (int, int) {
}

func (c *LayerX) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
}

func (c *LayerX) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
}

func (c *LayerX) ParamsUpdate (learning_rate float64) {
}
*/

type NeuralNetwork struct {
   Layers []Layer
}

func NewNeuralNetwork () *NeuralNetwork {
   n := new(NeuralNetwork)
   n.Layers = make([]Layer, 0)
   return n
}

func (n *NeuralNetwork) AddLayer (layer Layer) *NeuralNetwork {
   n.Layers = append(n.Layers, layer)
   return n
}

func (n *NeuralNetwork) Fit (input, expect *SimpleMatrix, learning_rate float64) *NeuralNetwork {
   m := len(n.Layers)
   X_next := input
   for i := 0; i < m; i++ {
      X_next = n.Layers[i].ForwardProp(X_next)
   }
   Y_pred := X_next
   grad_next := Y_pred.Add(expect, 1, -1)
   for i := m - 1; i >= 0; i-- {
      grad_next = n.Layers[i].BackwardProp(grad_next)
      n.Layers[i].ParamsUpdate(learning_rate)
   }
   return n
}

func (n *NeuralNetwork) Predict (input *SimpleMatrix) *SimpleMatrix {
   X_next := input
   for _, layer := range n.Layers {
      X_next = layer.ForwardProp(X_next)
   }
   return X_next
}

func (n *NeuralNetwork) Error (predict, expect *SimpleMatrix) float64 {
   return predict.Add(expect, 1, -1).Map(math.Abs).EltSum() / float64(predict.M * predict.N)
}
