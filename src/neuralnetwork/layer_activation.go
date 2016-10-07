package neuralnetwork

// ref: https://github.com/andersbll/nnet

type LayerActivation struct {
   LayerBase
   Fun, FunDerivative func (float64) float64
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
   default: /* "sigmoid" */
      c.Fun = Sigmoid
      c.FunDerivative = SigmoidDerivative
   }
   return c
}

func (c *LayerActivation) OutputDim () (int, int) {
   return c.lastInput.M, c.lastInput.N
}

func (c *LayerActivation) InputDim () (int, int) {
   return c.OutputDim()
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
