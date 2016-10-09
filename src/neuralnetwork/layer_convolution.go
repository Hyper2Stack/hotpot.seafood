package neuralnetwork

// ref: https://github.com/andersbll/nnet

type LayerConvolution struct {
   LayerBase
   W, B *SimpleMatrix
   DeltaWb []*SimpleMatrix
   WeightDecay float64
   InputM, M, N, ItemM, ItemN, KernelM, KernelN int
}

func NewLayerConvolution (
   input_m, input_n, output_n int,
   item_m, item_n, kernel_m, kernel_n int,
   weight_decay float64,
) *LayerConvolution {
   c := new(LayerConvolution)
   c.W = NewSimpleMatrix(
      input_n * kernel_m, output_n * kernel_n).FillRandom(-1, 1)
   c.B = NewSimpleMatrix(input_n, output_n)
   c.DeltaWb = make([]*SimpleMatrix, 2)
   c.DeltaWb[0] = NewSimpleMatrix(c.W.M, c.W.N) // dW
   c.DeltaWb[1] = NewSimpleMatrix(c.B.M, c.B.N) // db
   c.WeightDecay = weight_decay
   c.InputM = input_m
   c.M = input_n
   c.N = output_n
   c.ItemM = item_m
   c.ItemN = item_n
   c.KernelM = kernel_m
   c.KernelN = kernel_n
   return c
}

func (c *LayerConvolution) OutputDim () (int, int) {
   return c.InputM * c.ItemM, c.N * c.ItemN
}

func (c *LayerConvolution) InputDim () (int, int) {
   return c.InputM * c.ItemM, c.M * c.ItemN
}

func __layer_convolution_Badd__(a float64) func (float64) float64 {
   return func (x float64) float64 {
      return x + a
   }
}
func __layer_convolution_batch_conv__(
   out_m, out_n int,
   input *SimpleMatrix, item_m, item_n int,
   kernel *SimpleMatrix, kernel_m, kernel_n int,
   b *SimpleMatrix,
) *SimpleMatrix {
   R := NewSimpleMatrix(out_m, out_n)
   in_m := input.M / item_m
   in_n := input.N / item_n
   kr_n := kernel.N / kernel_n
   for i := in_m - 1; i >= 0; i-- {
      for k := kr_n - 1; k >= 0; k-- {
         conv := NewSimpleMatrix(item_m, item_n)
         for j := in_n - 1; j >= 0; j-- {
            conv = conv.Add(
               input.Window(
                  i * item_m, j * item_n, item_m, item_n,
               ).Convolute(
                  kernel.Window(
                     j * kernel_m, k * kernel_n, kernel_m, kernel_n),
               ),
               1, 1,
            )
         }
         // XW + b
         R = R.FillWindow(
            i * item_m, k * item_n,
            conv.Map(__layer_convolution_Badd__(b.Data[i][k])))
      }
   }
   return R
}
func (c *LayerConvolution) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = __layer_convolution_batch_conv__(
      c.InputM * c.ItemM, c.N * c.ItemN,
      input, c.ItemM, c.ItemN, c.W, c.KernelM, c.KernelN, c.B)
   return c.lastOutput.Clone()
}

func __layer_convolution_sum__(x, y float64) float64 {
   return x + y
}
func __layer_convolution_batch_backward__(
   out_m, out_n int, output_grad *SimpleMatrix,
   last_input *SimpleMatrix, item_m, item_n int,
   kernel *SimpleMatrix, kernel_m, kernel_n int,
) (*SimpleMatrix, *SimpleMatrix) {
   grad := NewSimpleMatrix(last_input.M, last_input.N)
   dW := NewSimpleMatrix(kernel.M, kernel.N)
   item_mask := NewSimpleMatrix(item_m, item_n)
   in_m := last_input.M / item_m
   in_n := last_input.N / item_n
   kr_n := kernel.N / kernel_n
   y_mid_offset := (kernel_m - 1) / 2
   x_mid_offset := (kernel_n - 1) / 2
   for i := in_m - 1; i >= 0; i-- {
      for k := kr_n - 1; k >= 0; k-- {
         for y := item_m - 1; y >= 0; y-- {
            for x := item_n - 1; x >= 0; x-- {
               scale := output_grad.Data[i * item_m + y][k * item_n + x]
               for j := in_n -1; j >= 0; j-- {
                  item_y := y - y_mid_offset
                  item_x := x - x_mid_offset
                  j_km := j * kernel_m
                  k_kn := k * kernel_n
                  i_im := i * item_m
                  j_in := j * item_n
                  item_mask.FillWindow(
                     item_y, item_x,
                     kernel.Window(j_km, k_kn, kernel_m, kernel_n).Scale(scale),
                  )
                  grad.FillWindowMap(
                     i_im + item_y, j_in + item_x,
                     item_mask.Window(item_y, item_x, kernel_m, kernel_n),
                     __layer_convolution_sum__,
                  )
                  dW.FillWindowMap(
                     j_km, k_kn,
                     last_input.Window(
                        i_im, j_in, item_m, item_n,
                     ).Window(
                        item_y, item_x, kernel_m, kernel_n,
                     ).Scale(scale),
                     __layer_convolution_sum__,
                  )
               }
            }
         }
      }
   }
   dW = dW.Scale(1.0 / float64(in_m))
   return grad, dW
}
func (c *LayerConvolution) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   item_m := c.ItemM
   item_n := c.ItemN
   c.lastGrad, c.DeltaWb[0] = __layer_convolution_batch_backward__(
      c.M * item_m, c.N * item_n, output_grad, c.lastInput, item_m, item_n, c.W, c.KernelM, c.KernelN)
   db := c.DeltaWb[1]
   for i := db.M - 1; i >= 0; i-- {
      for j := db.N - 1; j >= 0; j-- {
         db.Data[i][j] = output_grad.Window(
            i * item_m, j * item_n, item_m, item_n,
         ).EltSum() / float64(output_grad.M)
      }
   }
   c.DeltaWb[1] = db
   return c.lastGrad.Clone()
}

func (c *LayerConvolution) DeltaN () int {
   return 2
}

func (c *LayerConvolution) Delta () []*SimpleMatrix {
   return c.DeltaWb
}

func (c *LayerConvolution) CorrectDelta (delta []*SimpleMatrix, offset int) {
   c.DeltaWb[0] = delta[offset]
   c.DeltaWb[1] = delta[offset + 1]
}

func (c *LayerConvolution) ParamsUpdate (alpha float64) {
   c.W = c.W.Add(c.DeltaWb[0], 1 - c.WeightDecay, alpha)
   c.B = c.B.Add(c.DeltaWb[1], 1, alpha)
}
