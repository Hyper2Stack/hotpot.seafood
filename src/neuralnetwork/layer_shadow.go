package neuralnetwork

type LayerShadow struct {
   LayerBase
   Shadow Layer
}

func NewLayerShadow (shadow Layer) *LayerShadow {
   c := new(LayerShadow)
   c.Shadow = shadow
   return c
}

func (c *LayerShadow) LastInput () *SimpleMatrix {
   return c.Shadow.LastInput()
}

func (c *LayerShadow) LastOutput () *SimpleMatrix {
   return c.Shadow.LastOutput()
}

func (c *LayerShadow) LastGrad () *SimpleMatrix {
   return c.Shadow.LastGrad()
}

func (c *LayerShadow) Dim () (int, int) {
   return c.Shadow.Dim()
}

func (c *LayerShadow) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   return c.Shadow.ForwardProp(input)
}

func (c *LayerShadow) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   return c.Shadow.BackwardProp(output_grad)
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
