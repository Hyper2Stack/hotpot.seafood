package neuralnetwork

// ref: https://github.com/andersbll/nnet

type LayerLinear struct {
   Layer
   W, B *SimpleMatrix
   DeltaWb []*SimpleMatrix // W, b
   WeightScale, WeightDecay float64
   EnableB bool
   lastInput, lastOutput, lastGrad *SimpleMatrix
}

func NewLayerLinear (input_m, input_n, output_n int, weight_scale, weight_decay float64, enable_b bool) *LayerLinear {
   // weight_decay default: 0.0
   c := new(LayerLinear)
   c.W = NewSimpleMatrix(input_n, output_n).FillGuassian(0, weight_scale)
   c.B = NewSimpleMatrix(input_m, output_n)
   c.DeltaWb = make([]*SimpleMatrix, 2)
   c.DeltaWb[0] = NewSimpleMatrix(input_n, output_n) // dW
   c.DeltaWb[1] = NewSimpleMatrix(input_m, output_n) // db
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
   c.DeltaWb[0] = c.lastInput.T().Dot(output_grad).Add(c.W.Scale(c.WeightDecay), 1, 1)
   if c.EnableB {
      c.DeltaWb[1] = output_grad.Add(c.lastInput.Dot(c.DeltaWb[0]), 1, 1)
   }
   c.lastGrad = output_grad.Dot(c.W.T())
   return c.lastGrad.Clone()
}

func (c *LayerLinear) DeltaN () int {
   if c.EnableB {
      return 2
   }
   return 1
}

func (c *LayerLinear) Delta () []*SimpleMatrix {
   return c.DeltaWb
}

func (c *LayerLinear) CorrectDelta (delta []*SimpleMatrix, offset int) {
   c.DeltaWb[0] = delta[offset]
   if c.EnableB {
      c.DeltaWb[1] = delta[offset + 1]
   }
}

func (c *LayerLinear) ParamsUpdate (alpha float64) {
   c.W = c.W.Add(c.DeltaWb[0].Scale(alpha), 1, 1)
   if c.EnableB {
      c.B = c.B.Add(c.DeltaWb[1].Scale(alpha), 1, 1)
   }
}
