package neuralnetwork

type NeuralRecurrentChain struct {
   NeuralChain
}

func NewNeuralRecurrentChain (input_m, input_n int) *NeuralRecurrentChain {
   n := new(NeuralRecurrentChain)
   n.Layers = make([]Layer, 0)
   n.DefineInputDim(input_m, input_n)
   return n
}

func (n *NeuralRecurrentChain) PredictRestart () {
   for _, layer := range n.Layers {
      lrs := layer.(*LayerRecordShadow)
      lrs.action.ResetRecord(lrs)
   }
}

func (n *NeuralRecurrentChain) AddLayer (layer Layer) NeuralNetwork {
   return n.AddRecurrentLayer(layer, "input_record_delay_update")
}

func (n *NeuralRecurrentChain) AddRecurrentLayer (layer Layer, recurrence_type string) *NeuralRecurrentChain {
   var wrapper Layer
   switch recurrence_type {
   case "input_record":
      m, n := layer.InputDim()
      wrapper = NewLayerRecordShadow(layer, m, n, new(RecordInputOfLayerRecordShadow))
   case "output_record":
      m, n := layer.OutputDim()
      wrapper = NewLayerRecordShadow(layer, m, n, new(RecordOutputOfLayerRecordShadow))
   case "input_record_delay_update":
      m, n := layer.InputDim()
      wrapper = NewLayerRecordShadow(layer, m, n, new(RecordInputDelayUpdateOfLayerRecordShadow))
   case "output_record_delay_update":
      m, n := layer.OutputDim()
      wrapper = NewLayerRecordShadow(layer, m, n, new(RecordOutputDelayUpdateOfLayerRecordShadow))
   default: /* "basic_recurrence" */
      m, n := layer.OutputDim()
      wrapper = NewLayerRecordShadow(layer, m, n, new(RecurrenceOfLayerRecordShadow).Init(n, n))
   }
   n.Layers = append(n.Layers, wrapper)
   return n
}
