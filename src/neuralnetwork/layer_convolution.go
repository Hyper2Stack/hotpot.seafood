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

func __unused_lconv_i_conv__ (X, Kernel *SimpleMatrix) *SimpleMatrix {
   R := NewSimpleMatrix(X.M - Kernel.M + 1, X.N - Kernel.N + 1)
   // Kernel = Kernel.Flip180()
   m := Kernel.M - 1
   n := Kernel.N - 1
   K := NewSimpleMatrix(Kernel.M, Kernel.N)
   for i := m; i >= 0; i-- {
      for j := n; j >= 0; j-- {
         K.Data[i][j] = Kernel.Data[m - i][n - j]
      }
   }
   for i := X.M - Kernel.M; i >= 0; i-- {
      for j := X.N - Kernel.N; j >= 0; j-- {
         R.Data[i][j] = X.Window(i, j, Kernel.M, Kernel.N).EltMul(Kernel).EltSum()
      }
   }
   return R
}
func __lconv_matrix_conv__(
   inputs *SimpleMatrix, input_m, input_n, item_m, item_n int,
   kernels *SimpleMatrix, output_n, kernel_m, kernel_n int,
   b *SimpleMatrix,
) *SimpleMatrix {
   R := NewSimpleMatrix(input_m * item_m, output_n * item_n)
   fil_mid_h := kernel_m / 2
   fil_mid_w := kernel_n / 2
   for i := input_m - 1; i >= 0; i-- {
      for k := output_n - 1; k >= 0; k-- {
         for y := item_m - 1; y >= 0; y-- {
            y_off_min := -y
            if y_off_min < -fil_mid_h { y_off_min = -fil_mid_h }
            y_off_max := item_m-y
            if y_off_max > fil_mid_h+1 { y_off_max = fil_mid_h+1 }
            for x := item_n - 1; x >= 0; x-- {
               val := 0.0
               x_off_min:= -x
               if x_off_min < -fil_mid_w { x_off_min = -fil_mid_w }
               x_off_max:= item_n-x
               if x_off_max > fil_mid_w+1 { x_off_max = fil_mid_w+1 }
               for y_off := y_off_min; y_off < y_off_max; y_off++ {
                  for x_off := x_off_min; x_off < x_off_max; x_off++ {
                     item_y := y + y_off
                     item_x := x + x_off
                     k_y := fil_mid_h + y_off
                     k_x := fil_mid_w + x_off
                     for j := input_n - 1; j >= 0; j-- {
                        val += inputs.Data[i * item_m + item_y][j * item_n + item_x] * kernels.Data[j * kernel_m + k_y][k * kernel_n + k_x]
                     }
                  }
               }
               R.Data[i * item_m + y][k * item_n + x] = val
            }
         }
      }
   }
   return R
}
func (c *LayerConvolution) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   c.lastOutput = __lconv_matrix_conv__(
      input, c.InputM, c.M, c.ItemM, c.ItemN,
      c.W, c.N, c.KernelM, c.KernelN, c.B,
   )
   return c.lastOutput.Clone()
}


func __unused_lconv_b_conv__ (X, Kernel *SimpleMatrix) *SimpleMatrix {
   R := NewSimpleMatrix(X.M + Kernel.M - 1, X.N + Kernel.N - 1)
   km := Kernel.M - 1
   kn := Kernel.N - 1
   for i := X.M - 1; i >= -km; i-- {
      for j := X.N - 1; j >= -kn; j-- {
         R.Data[i + km][j + kn] = X.Window(i, j, km + 1, kn + 1).EltMul(Kernel).EltSum()
      }
   }
   return R
}
func __lconv_matrix_grad__ (
   last_inputs *SimpleMatrix, input_m, input_n, item_m, item_n int,
   kernels *SimpleMatrix, output_n, kernel_m, kernel_n int,
   grad *SimpleMatrix,
) (*SimpleMatrix, *SimpleMatrix) {
   dX := NewSimpleMatrix(last_inputs.M, last_inputs.N)
   dW := NewSimpleMatrix(kernels.M, kernels.N)
   fil_mid_h := kernel_m / 2
   fil_mid_w := kernel_n / 2
   for i := input_m - 1; i >= 0; i-- {
      for k := output_n - 1; k >= 0; k-- {
         for y := item_m - 1; y >= 0; y-- {
            y_off_min:= -y
            if y_off_min < -fil_mid_h { y_off_min = -fil_mid_h }
            y_off_max:= item_m-y
            if y_off_max > fil_mid_h+1 { y_off_max = fil_mid_h+1 }
            for x := item_n - 1; x >= 0; x-- {
               gradval := grad.Data[i * item_m + y][k * item_n + x]
               x_off_min:= -x
               if x_off_min < -fil_mid_w { x_off_min = -fil_mid_w }
               x_off_max:= item_n-x
               if x_off_max > fil_mid_w+1 { x_off_max = fil_mid_w+1 }
               for y_off := y_off_min; y_off < y_off_max; y_off++ {
                  for x_off := x_off_min; x_off < x_off_max; x_off++ {
                     item_y := y + y_off
                     item_x := x + x_off
                     k_y := fil_mid_h + y_off
                     k_x := fil_mid_w + x_off
                     for j := input_n - 1; j >= 0; j-- {
                        iIm := i * item_m
                        jIn := j * item_n
                        jKm := j * kernel_m
                        kKn := k * kernel_n
                        dX.Data[iIm + item_y][jIn + item_x] += kernels.Data[jKm + k_y][kKn + k_x] * gradval
                        dW.Data[jKm + k_y][kKn + k_x] += last_inputs.Data[iIm + item_y][jIn + item_x] * gradval
                     }
                  }
               }
            }
         }
      }
   }
   dW = dW.Scale(1.0/float64(input_m))
   return dX, dW
}
func (c *LayerConvolution) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   item_m := c.ItemM
   item_n := c.ItemN
   c.lastGrad, c.DeltaWb[0] = __lconv_matrix_grad__(
      c.lastInput, c.InputM, c.M, c.ItemM, c.ItemN,
      c.W, c.N, c.KernelM, c.KernelN, output_grad)
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
