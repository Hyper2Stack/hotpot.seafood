package neuralnetwork

import "math"

// ref: https://github.com/andersbll/nnet

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

func (c *LayerLogRegression) Delta () []*SimpleMatrix {
   return make([]*SimpleMatrix, 0)
}

func (c *LayerLogRegression) CorrectDelta (delta []*SimpleMatrix) {
}

func (c *LayerLogRegression) ParamsUpdate (alpha float64) {
}
