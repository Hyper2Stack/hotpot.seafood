package neuralnetwork

type ActionOfLayerRecordShadow interface {
   ResetRecord (layer *LayerRecordShadow)
   Record (layer *LayerRecordShadow)
   InputPlus (layer *LayerRecordShadow, input *SimpleMatrix) *SimpleMatrix
   GradPlus (layer *LayerRecordShadow, grad *SimpleMatrix) *SimpleMatrix
   DeltaUpdate (layer *LayerRecordShadow)
   DeltaApply (layer *LayerRecordShadow, alpha float64)
}

type LayerRecordShadow struct {
   LayerShadow
   cache []*SimpleMatrix
   cursor int
   recordM, recordN int
   action ActionOfLayerRecordShadow
}

func NewLayerRecordShadow (shadow Layer, record_m, record_n int, action ActionOfLayerRecordShadow) *LayerRecordShadow {
   c := new(LayerRecordShadow)
   c.Shadow = shadow
   c.recordM = record_m
   c.recordN = record_n
   c.action = action
   c.action.ResetRecord(c)
   return c
}

func (c *LayerRecordShadow) SwitchContext (i int) *LayerRecordShadow {
   if i < 0 || i >= len(c.cache) {
      return nil
   }
   c.cursor = i
   return c
}

func (c *LayerRecordShadow) MoveNext () *LayerRecordShadow {
   return c.SwitchContext(c.cursor + 1)
}

func (c *LayerRecordShadow) MovePrev () *LayerRecordShadow {
   return c.SwitchContext(c.cursor - 1)
}

func (c *LayerRecordShadow) Current () *SimpleMatrix {
   return c.cache[c.cursor]
}

func (c *LayerRecordShadow) Next () *SimpleMatrix {
   return c.cache[c.cursor + 1]
}

func (c *LayerRecordShadow) Prev () *SimpleMatrix {
   return c.cache[c.cursor - 1]
}

func (c *LayerRecordShadow) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   input = c.action.InputPlus(c, input)
   output := c.Shadow.ForwardProp(input)
   c.action.Record(c)
   return output
}

func (c *LayerRecordShadow) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   output_grad = c.action.GradPlus(c, output_grad)
   delta := c.Shadow.BackwardProp(output_grad)
   c.action.DeltaUpdate(c)
   c.MovePrev()
   return delta
}

func (c *LayerRecordShadow) ParamsUpdate (alpha float64) {
   if c.cursor > 0 {
      return
   }
   // record and move cursor forward
   // do backward procedure to move back
   // finally cursor should be at the head
   // then update params with aggregated delta
   c.action.DeltaApply(c, alpha)
   c.Shadow.ParamsUpdate(alpha)
   c.action.ResetRecord(c)
}


type NopActionOfLayerRecordShadow struct {
   ActionOfLayerRecordShadow
}

func (a *NopActionOfLayerRecordShadow) ResetRecord (c *LayerRecordShadow) {
   c.cache = make([]*SimpleMatrix, 1)
   c.cache[0] = NewSimpleMatrix(c.recordM, c.recordN)
   c.cursor = 0
}

func (a *NopActionOfLayerRecordShadow) Record (c *LayerRecordShadow) {
   c.cache = append(c.cache, nil)
   c.MoveNext()
}

func (a *NopActionOfLayerRecordShadow) InputPlus (c *LayerRecordShadow, input *SimpleMatrix) *SimpleMatrix {
   return input
}

func (a *NopActionOfLayerRecordShadow) GradPlus (c *LayerRecordShadow, grad *SimpleMatrix) *SimpleMatrix {
   return grad
}

func (a *NopActionOfLayerRecordShadow) DeltaUpdate (c *LayerRecordShadow) {
}

func (a *NopActionOfLayerRecordShadow) DeltaApply (c *LayerRecordShadow, alpha float64) {
}


type RecordInputOfLayerRecordShadow struct {
   NopActionOfLayerRecordShadow
}

func (a *RecordInputOfLayerRecordShadow) GradPlus (c *LayerRecordShadow, grad *SimpleMatrix) *SimpleMatrix {
   c.LoadLastInput(c.Current())
   return grad
}

func (a *RecordInputOfLayerRecordShadow) Record (c *LayerRecordShadow) {
   c.cache = append(c.cache, c.LastInput())
   c.MoveNext()
}


type RecordOutputOfLayerRecordShadow struct {
   NopActionOfLayerRecordShadow
}

func (a *RecordOutputOfLayerRecordShadow) GradPlus (c *LayerRecordShadow, grad *SimpleMatrix) *SimpleMatrix {
   c.LoadLastOutput(c.Current())
   return grad
}

func (a *RecordOutputOfLayerRecordShadow) Record (c *LayerRecordShadow) {
   c.cache = append(c.cache, c.LastOutput())
   c.MoveNext()
}
