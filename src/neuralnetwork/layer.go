package neuralnetwork

type CacheLayer interface {
   LastInput () *SimpleMatrix
   LastOutput () *SimpleMatrix
   LastGrad () *SimpleMatrix
}

type Layer interface {
   CacheLayer
   // Setup layer with parameters that are unknown at 
   Setup ()
   // Output matrix shape M * N
   Dim () (int, int)
   // Calculate layer output for given input (forward propagation).
   ForwardProp (input *SimpleMatrix) *SimpleMatrix
   // Calculate input gradient.
   BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix
   // Get/Set Learn Delta
   Delta () []*SimpleMatrix
   CorrectDelta (delta []*SimpleMatrix)
   // Update layer parameter gradients as calculated from BackwardProp(). 
   ParamsUpdate (alpha float64)
}

type LossMixin interface {
   // Calculate mean loss given output and predicted output.
   Loss (output, output_pred *SimpleMatrix) *SimpleMatrix
   // Calculate input gradient given output and predicted output.
   //InputGrad (output, output_pred *SimpleMatrix) *SimpleMatrix
}

/* Layer Template
type LayerX struct {
   Layer
   lastInput, lastOutput, lastGrad *SimpleMatrix
}

func NewLayerX (input_m, input_n, output_n int) *LayerX {
   c := new(LayerX)
   return c
}

func (c *LayerX) LastInput () *SimpleMatrix {
   return c.lastInput
}

func (c *LayerX) LastOutput () *SimpleMatrix {
   return c.lastOutput
}

func (c *LayerX) LastGrad () *SimpleMatrix {
   return c.lastGrad
}

func (c *LayerX) Setup () {
}

func (c *LayerX) Dim () (int, int) {
}

func (c *LayerX) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = 
   return c.lastOutput.Clone()
}

func (c *LayerX) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad =
   return c.lastGrad.Clone()
}

func (c *LayerX) Delta () []*SimpleMatrix {
   return make([]*SimpleMatrix, 0)
}

func (c *LayerX) CorrectDelta (delta []*SimpleMatrix) {
}

func (c *LayerX) ParamsUpdate (alpha float64) {
}
*/
