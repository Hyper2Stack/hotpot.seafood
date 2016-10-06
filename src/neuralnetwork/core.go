package neuralnetwork

import "math"

// ref: https://github.com/andersbll/nnet

type CacheLayer interface {
   LastInput () *SimpleMatrix
   LastOutput () *SimpleMatrix
   LastGrad () *SimpleMatrix
}

type Layer interface {
   CacheLayer
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
   WeightScale, WeightDecay float64
   EnableB bool
   lastInput, lastOutput, lastGrad *SimpleMatrix
}

func NewLayerLinear (input_m, input_n, output_n int, weight_scale, weight_decay float64, enable_b bool) *LayerLinear {
   // weight_decay default: 0.0
   c := new(LayerLinear)
   c.W = NewSimpleMatrix(input_n, output_n).FillGuassian(0, weight_scale)
   c.B = NewSimpleMatrix(input_m, output_n)
   c.WeightScale = weight_scale
   c.WeightDecay = weight_decay
   return c
}

func (c *LayerLinear) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerLinear) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerLinear) LastGrad () *SimpleMatrix {
   return c.lastGrad
}

func (c *LayerLinear) Setup () {
}

func (c *LayerLinear) Dim () (int, int) {
   return c.B.M, c.B.N
}

func (c *LayerLinear) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = input.Dot(c.W)
   if c.EnableB {
      c.lastOutput = c.lastOutput.Add(c.B, 1, 1)
   }
   return c.lastOutput.Clone()
}

func (c *LayerLinear) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.DeltaW = c.lastInput.T().Dot(output_grad).Add(c.W.Scale(c.WeightDecay), 1, 1)
   if c.EnableB {
      c.DeltaB = output_grad.Add(c.lastInput.Dot(c.DeltaW), 1, 1)
   }
   c.lastGrad = output_grad.Dot(c.W.T())
   return c.lastGrad.Clone()
}

func (c *LayerLinear) ParamsUpdate (learning_rate float64) {
   c.W = c.W.Add(c.DeltaW.Scale(learning_rate), 1, 1)
   if c.EnableB {
      c.B = c.B.Add(c.DeltaW.Scale(learning_rate), 1, 1)
   }
}


type LayerActivation struct {
   Layer
   Fun, FunDerivative func (float64) float64
   lastInput, lastOutput, lastGrad *SimpleMatrix
}

func NewLayerActivation (input_m, input_n int, fun_type string) *LayerActivation {
   c := new(LayerActivation)
   c.lastInput = NewSimpleMatrix(input_m, input_n)
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

func (c *LayerActivation) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerActivation) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerActivation) LastGrad () *SimpleMatrix {
   return c.lastGrad
}

func (c *LayerActivation) Setup () {
}

func (c *LayerActivation) Dim () (int, int) {
   return c.lastInput.M, c.lastInput.N
}

func (c *LayerActivation) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = input.Map(c.Fun)
   return c.lastOutput.Clone()
}

func (c *LayerActivation) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad = output_grad.EltMul(c.lastOutput.Map(c.FunDerivative))
   return c.lastGrad.Clone()
}

func (c *LayerActivation) ParamsUpdate (learning_rate float64) {
}


type LayerLogRegression struct {
   Layer
   LossMixin
   M, N int
   lastInput, lastOutput *SimpleMatrix
}

func NewLayerLogRegression (input_m, input_n int) *LayerLogRegression {
   c := new(LayerLogRegression)
   c.M = input_m
   c.N = input_n
   return c
}

func (c *LayerLogRegression) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerLogRegression) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerLogRegression) LastGrad () *SimpleMatrix {
   return nil
}

func (c *LayerLogRegression) Setup () {
}

func (c *LayerLogRegression) Dim () (int, int) {
   return c.M, c.N
}

func (c *LayerLogRegression) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = input.Softmax()
   return c.lastOutput.Clone()
}

func (c *LayerLogRegression) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   // LogRegression does not support back-propagation of gradients.
   // It should occur only as the last layer of a NeuralChain.
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
   lastInput, lastOutput, lastGrad *SimpleMatrix
}

func NewLayerX (input_m, input_n, output_n int) *LayerX {
   c := new(LayerX)
   return c
}

func (c *LayerX) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerX) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerX) LastGrad () *SimpleMatrix {
   return c.lastGrad
}

func (c *LayerX) Setup () {
}

func (c *LayerX) Dim () (int, int) {
}

func (c *LayerX) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = 
   return c.lastOutput.Clone()
}

func (c *LayerX) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad =
   return c.lastGrad.Clone()
}

func (c *LayerX) ParamsUpdate (learning_rate float64) {
}
*/

type NeuralChain struct {
   Layers []Layer
}

func NewNeuralChain () *NeuralChain {
   n := new(NeuralChain)
   n.Layers = make([]Layer, 0)
   return n
}

func (n *NeuralChain) AddLayer (layer Layer) *NeuralChain {
   n.Layers = append(n.Layers, layer)
   return n
}

func (n *NeuralChain) Fit (input, expect *SimpleMatrix, alpha float64) *NeuralChain {
   n.Learn(n.Predict(input), expect).Update(alpha)
   return n
}

func (n *NeuralChain) Predict (input *SimpleMatrix) *SimpleMatrix {
   X_next := input
   for _, layer := range n.Layers {
      X_next = layer.ForwardProp(X_next)
   }
   return X_next
}

func (n *NeuralChain) Learn (predict *SimpleMatrix, expect *SimpleMatrix) *NeuralChain {
   m := len(n.Layers)
   grad_next := expect.Add(predict, 1, -1)
   for i := m - 1; i >= 0; i-- {
      grad_next = n.Layers[i].BackwardProp(grad_next)
   }
   return n
}

func (n *NeuralChain) Update (alpha float64) *NeuralChain {
   m := len(n.Layers)
   for i := m - 1; i >= 0; i-- {
      n.Layers[i].ParamsUpdate(alpha)
   }
   return n
}

func (n *NeuralChain) Error (predict, expect *SimpleMatrix) float64 {
   return predict.Add(expect, 1, -1).Map(math.Abs).EltSum() / float64(predict.M * predict.N)
}
