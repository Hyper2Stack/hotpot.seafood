package neuralnetwork

// ref: https://github.com/andersbll/nnet

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

func (c *LayerActivation) Delta () []*SimpleMatrix {
   return make([]*SimpleMatrix, 0)
}

func (c *LayerActivation) CorrectDelta (delta []*SimpleMatrix) {
}

func (c *LayerActivation) ParamsUpdate (alpha float64) {
}
