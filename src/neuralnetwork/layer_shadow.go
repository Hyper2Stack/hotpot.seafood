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

func (c *LayerShadow) LoadLastInput (input *SimpleMatrix) {
   c.Shadow.LoadLastInput(input)
}

func (c *LayerShadow) LoadLastOutput (output *SimpleMatrix) {
   c.Shadow.LoadLastOutput(output)
}

func (c *LayerShadow) OutputDim () (int, int) {
   return c.Shadow.OutputDim()
}

func (c *LayerShadow) InputDim () (int, int) {
   return c.Shadow.InputDim()
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
