package neuralnetwork

type LayerSelfishShadow struct {
   LayerBase
   Shadow Layer
}

func NewLayerSelfishShadow (shadow Layer) *LayerSelfishShadow {
   c := new(LayerSelfishShadow)
   c.Shadow = shadow
   return c
}

func (c *LayerSelfishShadow) Dim () (int, int) {
   return c.Shadow.Dim()
}

func (c *LayerSelfishShadow) Activate (copy_input, copy_output, copy_grad bool) *LayerSelfishShadow {
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

func (c *LayerSelfishShadow) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = c.Shadow.ForwardProp(input).Clone()
   return c.lastOutput.Clone()
}

func (c *LayerSelfishShadow) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad = c.Shadow.BackwardProp(output_grad).Clone()
   return c.lastGrad.Clone()
}

func (c *LayerSelfishShadow) DeltaN () int {
   return c.Shadow.DeltaN()
}

func (c *LayerSelfishShadow) Delta () []*SimpleMatrix {
   return c.Shadow.Delta()
}

func (c *LayerSelfishShadow) CorrectDelta (delta []*SimpleMatrix, offset int) {
   c.Shadow.CorrectDelta(delta, offset)
}

func (c *LayerSelfishShadow) ParamsUpdate (alpha float64) {
   c.Shadow.ParamsUpdate(alpha)
}
