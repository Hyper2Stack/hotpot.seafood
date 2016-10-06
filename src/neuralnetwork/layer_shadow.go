package neuralnetwork

type LayerShadow struct {
   Layer
   lastInput, lastOutput, lastGrad *SimpleMatrix
   Shadow Layer
}

func NewLayerShadow (shadow Layer) *LayerShadow {
   c := new(LayerShadow)
   c.Shadow = shadow
   return c
}

func (c *LayerShadow) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerShadow) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerShadow) LastGrad () *SimpleMatrix {
   return c.lastGrad
}

func (c *LayerShadow) Setup () {
}

func (c *LayerShadow) Dim () (int, int) {
   return c.Shadow.Dim()
}

func (c *LayerShadow) Activate (copy_input, copy_output, copy_grad bool) *LayerShadow {
   if copy_input {
      origin := c.Shadow.LastInput()
      if c.lastInput != (*SimpleMatrix)(nil) && origin != (*SimpleMatrix)(nil) {
         origin.FillWindow(0, 0, c.lastInput)
      }
   }
   if copy_output {
      origin := c.Shadow.LastOutput()
      if c.lastOutput != (*SimpleMatrix)(nil) && origin != (*SimpleMatrix)(nil) {
         origin.FillWindow(0, 0, c.lastOutput)
      }
   }
   if copy_grad {
      origin := c.Shadow.LastGrad()
      if c.lastGrad != (*SimpleMatrix)(nil) && origin != (*SimpleMatrix)(nil) {
         origin.FillWindow(0, 0, c.lastGrad)
      }
   }
   return c
}

func (c *LayerShadow) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = c.Shadow.ForwardProp(input).Clone()
   return c.lastOutput.Clone()
}

func (c *LayerShadow) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad = c.Shadow.BackwardProp(output_grad).Clone()
   return c.lastGrad.Clone()
}

func (c *LayerShadow) DeltaN () int {
   return c.Shadow.DeltaN()
}

func (c *LayerShadow) Delta () []*SimpleMatrix {
   return c.Shadow.Delta()
}

func (c *LayerShadow) CorrectDelta (delta []*SimpleMatrix, offset int) {
   c.Shadow.CorrectDelta(delta, offset)
}

func (c *LayerShadow) ParamsUpdate (alpha float64) {
   c.Shadow.ParamsUpdate(alpha)
}
