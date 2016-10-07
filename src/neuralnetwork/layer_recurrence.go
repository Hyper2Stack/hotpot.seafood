package neuralnetwork

type RecurrenceOfLayerRecordShadow struct {
   RecordOutputOfLayerRecordShadow
   H, DeltaH *SimpleMatrix
   lastDelta *SimpleMatrix
}

func (a *RecurrenceOfLayerRecordShadow) Init (record_m, record_n int) *RecurrenceOfLayerRecordShadow {
   a.H = NewSimpleMatrix(record_m, record_n).FillRandom(-1, 1)
   return a
}

func (a *RecurrenceOfLayerRecordShadow) InitFill (h *SimpleMatrix) *RecurrenceOfLayerRecordShadow {
   a.H = h
   return a
}

func (a *RecurrenceOfLayerRecordShadow) ResetRecord (c *LayerRecordShadow) {
   c.cache = make([]*SimpleMatrix, 1)
   c.cache[0] = NewSimpleMatrix(c.recordM, c.recordN)
   c.cursor = 0
   a.lastDelta = NewSimpleMatrix(c.recordM, c.recordN)
   a.DeltaH = NewSimpleMatrix(c.recordN, c.recordN)
}

func (a *RecurrenceOfLayerRecordShadow) InputPlus (c *LayerRecordShadow, input *SimpleMatrix) *SimpleMatrix {
   return input.Add(c.Current().Dot(a.H), 1, 1)
}

func (a *RecurrenceOfLayerRecordShadow) GradPlus (c *LayerRecordShadow, grad *SimpleMatrix) *SimpleMatrix {
   c.LoadLastOutput(c.Current())
   return grad.Add(a.lastDelta.Dot(a.H.T()), 1, 1)
}

func (a *RecurrenceOfLayerRecordShadow) DeltaUpdate (c *LayerRecordShadow) {
   a.lastDelta = c.LastGrad()
   a.DeltaH = a.DeltaH.Add(c.Prev().T().Dot(a.lastDelta), 1, 1)
}

func (a *RecurrenceOfLayerRecordShadow) DeltaApply (c *LayerRecordShadow, alpha float64) {
   a.H = a.H.Add(a.DeltaH, 1, alpha)
}


type RecordOutputDelayUpdateOfLayerRecordShadow struct {
   RecordOutputOfLayerRecordShadow
   Delta []*SimpleMatrix
}

func (a *RecordOutputDelayUpdateOfLayerRecordShadow) DeltaUpdate (c *LayerRecordShadow) {
   if a.Delta == ([]*SimpleMatrix)(nil) {
      a.Delta = c.Delta()
   } else {
      delta := c.Delta()
      for i, d := range delta {
         a.Delta[i] = a.Delta[i].Add(d, 1, 1)
      }
   }
}

func (a *RecordOutputDelayUpdateOfLayerRecordShadow) DeltaApply (c *LayerRecordShadow, alpha float64) {
   c.CorrectDelta(a.Delta, 0)
}


type RecordInputDelayUpdateOfLayerRecordShadow struct{
   RecordInputOfLayerRecordShadow
   Delta []*SimpleMatrix
}

func (a *RecordInputDelayUpdateOfLayerRecordShadow) DeltaUpdate (c *LayerRecordShadow) {
   if a.Delta == ([]*SimpleMatrix)(nil) {
      a.Delta = c.Delta()
   } else {
      delta := c.Delta()
      for i, d := range delta {
         a.Delta[i] = a.Delta[i].Add(d, 1, 1)
      }
   }
}

func (a *RecordInputDelayUpdateOfLayerRecordShadow) DeltaApply (c *LayerRecordShadow, alpha float64) {
   c.CorrectDelta(a.Delta, 0)
}
