package neuralnetwork

// ref: https://github.com/andersbll/nnet
import "math"

type LayerPoolMax struct {
   LayerBase
   lastContribution *SimpleMatrix
   InputM, InputN, ItemM, ItemN int
   M, N, PoolM, PoolN int
}

func NewLayerPoolMax (input_m, input_n, item_n, item_m, pool_m, pool_n int) *LayerPoolMax {
   c := new(LayerPoolMax)
   c.lastContribution = NewSimpleMatrix(input_m * item_m, input_n * item_n)
   c.InputM = input_m
   c.InputN = input_n
   c.ItemM = item_m
   c.ItemN = item_n
   c.M = input_m
   c.N = input_n
   c.PoolM = pool_m
   c.PoolN = pool_n
   return c
}

func (c *LayerPoolMax) OutputDim () (int, int) {
   m := c.M * c.ItemM
   n := c.N * c.ItemN
   if m % c.PoolM == 0 {
      m = m / c.PoolM
   } else {
      m = m / c.PoolM + 1
   }
   if n % c.PoolN == 0 {
      n = n / c.PoolN
   } else {
      n = n / c.PoolN + 1
   }
   return m, n
}

func (c *LayerPoolMax) InputDim () (int, int) {
   return c.InputM * c.ItemM, c.InputN * c.ItemN
}

func __layer_pool_batch_poolmax__(
   out_m, out_n int,
   input *SimpleMatrix, item_m, item_n int,
   pool_m, pool_n int,
   contrib *SimpleMatrix,
) *SimpleMatrix {
   R := NewSimpleMatrix(out_m, out_n)
   input_m := input.M / item_m
   if input.M % item_m > 0 {
      input_m ++
   }
   input_n := input.N / item_n
   if input.N % item_n > 0 {
      input_n ++
   }
   pool_item_m := out_m / input_m
   pool_item_n := out_n / input_n
   for i := 0; i < input_m; i++ {
      for j := 0; j < input_n; j++ {
         item_i := i * item_m
         item_j := j * item_n
         item := input.Window(item_i, item_j, item_m, item_n)
         pool := item.Pool(
            pool_m, pool_n, pool_m, pool_n, math.Max, math.Inf(-1))
         R.FillWindow(i * pool_item_m, j * pool_item_n, pool)
         for p := pool_item_m - 1; p >= 0; p-- {
            for q := pool_item_n - 1; q >= 0; q-- {
               max := pool.Data[p][q]
               pi := item_i + p * pool_m
               qj := item_j + q * pool_n
               for y := pool_m - 1; y >= 0; y-- {
                  for x := pool_n - 1; x >= 0; x-- {
                     if input.Data[pi + y][qj + x] == max {
                        contrib.Data[pi + y][qj + x] = 1
                     } else {
                        contrib.Data[pi + y][qj + x] = 0
                     }
                  }
               }
            }
         }
      }
   }
   return R
}
func (c *LayerPoolMax) ForwardProp (input *SimpleMatrix) *SimpleMatrix {
   c.lastInput = input.Clone()
   out_m, out_n := c.OutputDim()
   c.lastOutput = __layer_pool_batch_poolmax__(
      out_m, out_n, input, c.ItemM, c.ItemN,
      c.PoolM, c.PoolN, c.lastContribution)
   return c.lastOutput.Clone()
}

func __layer_pool_batch_maxbackward__(
   input, grad *SimpleMatrix, item_m, item_n int,
   pool_m, pool_n int,
   contrib *SimpleMatrix,
) *SimpleMatrix {
   in_m := input.M
   in_n := input.N
   R := NewSimpleMatrix(in_m, in_n)
   for i := grad.M / pool_m - 1; i >= 0; i -- {
      for j := grad.N / pool_n - 1; j >= 0; j -- {
         for p := pool_m - 1; p >= 0; p-- {
            for q := pool_n - 1; q >= 0; q-- {
               input_i := i * item_m + p * pool_m
               input_j := j * item_n + q * pool_n
               pool_contrib := contrib.Window(input_i, input_j, pool_m, pool_n)
               pool_contrib_sum := pool_contrib.EltSum()
               if pool_contrib_sum <= 1 {
                  pool_contrib_sum = 1
               }
               pool_contrib = pool_contrib.Scale(grad.Data[i * pool_m + p][j * pool_n + q] / pool_contrib_sum)
               R.FillWindow(input_i, input_j, pool_contrib)
            }
         }
      }
   }
   return R
}
func (c *LayerPoolMax) BackwardProp (output_grad *SimpleMatrix) *SimpleMatrix {
   c.lastGrad = __layer_pool_batch_maxbackward__(
      c.lastInput, output_grad, c.ItemM, c.ItemN,
      c.PoolM, c.PoolN, c.lastContribution)
   return c.lastGrad
}

func (c *LayerPoolMax) DeltaN () int {
   return 0
}

func (c *LayerPoolMax) Delta () []*SimpleMatrix {
   r := make([]*SimpleMatrix, 0)
   return r
}

func (c *LayerPoolMax) CorrectDelta (delta []*SimpleMatrix, offset int) {
}

func (c *LayerPoolMax) ParamsUpdate (alpha float64) {
}
