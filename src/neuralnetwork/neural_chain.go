package neuralnetwork

import "math"

type NeuralChain struct {
   LayerBase
   NeuralNetwork
   Layers []Layer
   InputM, InputN int
}

func NewNeuralChain () *NeuralChain {
   n := new(NeuralChain)
   n.Layers = make([]Layer, 0)
   return n
}

func (n *NeuralChain) AddLayer (layer Layer) NeuralNetwork {
   n.Layers = append(n.Layers, layer)
   return n
}

func (n *NeuralChain) Fit (input, expect *SimpleMatrix, alpha float64) NeuralNetwork {
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

func (n *NeuralChain) Learn (predict *SimpleMatrix, expect *SimpleMatrix) NeuralNetwork {
   m := len(n.Layers)
   grad_next := expect.Add(predict, 1, -1)
   for i := m - 1; i >= 0; i-- {
      grad_next = n.Layers[i].BackwardProp(grad_next)
   }
   return n
}

func (n *NeuralChain) Update (alpha float64) NeuralNetwork {
   m := len(n.Layers)
   for i := m - 1; i >= 0; i-- {
      n.Layers[i].ParamsUpdate(alpha)
   }
   return n
}

func (n *NeuralChain) Error (predict, expect *SimpleMatrix) float64 {
   return expect.Add(predict, 1, -1).Map(math.Abs).EltSum() / float64(predict.M * predict.N)
}

func (c *NeuralChain) OutputDim () (int, int) {
   n := len(c.Layers)
   if n == 0 {
      return 0, 0
   }
   return c.Layers[n - 1].OutputDim()
}

func (c *NeuralChain) DefineInputDim (m, n int) *NeuralChain {
   c.InputM = m
   c.InputN = n
   return c
}

func (c *NeuralChain) InputDim () (int, int) {
   return c.InputM, c.InputN
}

func (c *NeuralChain) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = c.Predict(c.lastInput)
   return c.lastOutput.Clone()
}

func (c *NeuralChain) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   n := len(c.Layers)
   if n == 0 {
      return NewSimpleMatrix(0, 0)
   }
   c.Learn(c.lastOutput, c.lastOutput.Add(output_grad, 1, 1))
   c.lastGrad = c.Layers[0].LastGrad()
   return c.lastGrad.Clone()
}

func (c *NeuralChain) DeltaN () int {
   n := 0
   for _, layer := range c.Layers {
      n += layer.DeltaN()
   }
   return n
}

func (c *NeuralChain) Delta () []*SimpleMatrix {
   r := make([]*SimpleMatrix, 0)
   for _, layer := range c.Layers {
      if layer.DeltaN() == 0 {
         continue
      }
      r = append(r, layer.Delta() ...)
   }
   return make([]*SimpleMatrix, 0)
}

func (c *NeuralChain) CorrectLayerDelta (delta []*SimpleMatrix, offset, layerIndex int) {
   c.Layers[layerIndex].CorrectDelta(delta, offset)
}

func (c *NeuralChain) CorrectDelta (delta []*SimpleMatrix, offset int) {
   for i, layer := range c.Layers {
      c.CorrectLayerDelta(delta, offset, i)
      offset += layer.DeltaN()
   }
}

func (c *NeuralChain) ParamsUpdate (alpha float64) {
   c.Update(alpha)
}
