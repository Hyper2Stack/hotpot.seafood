package neuralnetwork

// ref: https://github.com/andersbll/nnet

type LayerFlatten struct {
   LayerBase
   InputM, InputN int
}

func NewLayerFlatten (input_m, input_n int) *LayerFlatten {
   c := new(LayerFlatten)
   c.InputM = input_m
   c.InputN = input_n
   return c
}

func (c *LayerFlatten) OutputDim () (int, int) {
   return 1, c.InputM * c.InputN
}

func (c *LayerFlatten) InputDim () (int, int) {
   return c.InputM, c.InputN
}

func (c *LayerFlatten) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   return input.Reshape(1, c.InputM * c.InputN)
}

func (c *LayerFlatten) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   return output_grad.Reshape(c.InputM, c.InputN)
}

func (c *LayerFlatten) DeltaN () int {
   return 0
}

func (c *LayerFlatten) Delta () []*SimpleMatrix {
   return make([]*SimpleMatrix, 0)
}

func (c *LayerFlatten) CorrectDelta (delta []*SimpleMatrix, offset int) {
}

func (c *LayerFlatten) ParamsUpdate (alpha float64) {
}
