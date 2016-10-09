package neuralnetwork

import (
   "fmt"
   "math"
   "math/rand"
)

type SimpleMatrix struct {
   M, N int
   Data [][]float64
}


func NewSimpleMatrix (m, n int) *SimpleMatrix {
   mat := new(SimpleMatrix)
   mat.M = m
   mat.N = n
   mat.Data = make([][]float64, m)
   for i := m - 1; i >= 0; i-- {
      mat.Data[i] = make([]float64, n)
   }
   return mat
}

func (X *SimpleMatrix) Dump () {
   fmt.Println("-- matrix ---------------------")
   for _, row := range X.Data {
      fmt.Println(row)
   }
   fmt.Println("===============================")
}

func (X *SimpleMatrix) T() *SimpleMatrix {
   R := NewSimpleMatrix(X.N, X.M)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[j][i] = X.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) Fill (x float64) *SimpleMatrix {
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         X.Data[i][j] = x
      }
   }
   return X
}

func (X *SimpleMatrix) FillElt (x []float64) *SimpleMatrix {
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         X.Data[i][j] = x[i * X.N + j]
      }
   }
   return X
}

func (X *SimpleMatrix) FillRandom (a, b float64) *SimpleMatrix {
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         X.Data[i][j] = rand.Float64() * (b - a) + a
      }
   }
   return X
}

func (X *SimpleMatrix) FillGuassian (mu, std float64) *SimpleMatrix {
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         X.Data[i][j] = RandomGuassian(mu, std)
      }
   }
   return X
}

func (X *SimpleMatrix) FillWindow (y, x int, Y *SimpleMatrix) *SimpleMatrix {
   m := y + Y.M
   if m > X.M {
      m = X.M
   }
   n := x + Y.N
   if n > X.N {
      n = X.N
   }
   for i := m - 1; i >= y; i-- {
      if i < 0 {
         continue
      }
      for j := n - 1; j >= x; j-- {
         if j < 0 {
            continue
         }
         X.Data[i][j] = Y.Data[i - y][j - x]
      }
   }
   return X
}

func (X *SimpleMatrix) FillWindowMap (
   y, x int, Y *SimpleMatrix, f func (float64, float64) float64,
) *SimpleMatrix {
   m := y + Y.M
   if m > X.M {
      m = X.M
   }
   n := x + Y.N
   if n > X.N {
      n = X.N
   }
   for i := m - 1; i >= y; i-- {
      if i < 0 {
         continue
      }
      for j := n - 1; j >= x; j-- {
         if j < 0 {
            continue
         }
         X.Data[i][j] = f(X.Data[i][j], Y.Data[i - y][j - x])
      }
   }
   return X
}

func (X *SimpleMatrix) Clone () *SimpleMatrix {
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) Reshape (m, n int) *SimpleMatrix {
   if m * n != X.M * X.N {
      return nil
   }
   R := NewSimpleMatrix(m, n)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         k := i * X.N + j
         R.Data[k / n][k % n] = X.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) ConnectRight (Y *SimpleMatrix) *SimpleMatrix {
   if X.M != Y.M {
      return nil
   }
   R := NewSimpleMatrix(X.M, X.N + Y.N)
   R.FillWindow(0, 0, X)
   R.FillWindow(0, X.N, Y)
   return R
}

func (X *SimpleMatrix) ConnectLeft (Y *SimpleMatrix) *SimpleMatrix {
   if X.M != Y.M {
      return nil
   }
   R := NewSimpleMatrix(X.M, X.N + Y.N)
   R.FillWindow(0, 0, Y)
   R.FillWindow(0, Y.N, X)
   return R
}

func (X *SimpleMatrix) ConnectBottom (Y *SimpleMatrix) *SimpleMatrix {
   if X.N != Y.N {
      return nil
   }
   R := NewSimpleMatrix(X.M + Y.M, X.N)
   R.FillWindow(0, 0, X)
   R.FillWindow(X.M, 0, Y)
   return R
}

func (X *SimpleMatrix) ConnectTop (Y *SimpleMatrix) *SimpleMatrix {
   if X.N != Y.N {
      return nil
   }
   R := NewSimpleMatrix(X.M + Y.M, X.N)
   R.FillWindow(0, 0, Y)
   R.FillWindow(Y.M, 0, X)
   return R
}

func (X *SimpleMatrix) Softmax () *SimpleMatrix {
   R      := NewSimpleMatrix(X.M, X.N)
   maxval := math.Inf(-1)
   scale  := 0.0
   // equals: maxval := X.EltMax()
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         if X.Data[i][j] > maxval {
            maxval = X.Data[i][j]
         }
      }
   }
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = math.Exp(X.Data[i][j] - maxval)
         scale += R.Data[i][j]
      }
   }
   // equals: return R.Scale(1.0 / scale)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] /= scale
      }
   }
   return R
}

func (X *SimpleMatrix) Convolute (Kernel *SimpleMatrix) *SimpleMatrix {
   R := NewSimpleMatrix(X.M, X.N)
   // XXX Kernel P * Q, P, Q = 0 (mod 2) issue
   m_mid_offset := (Kernel.M - 1) / 2
   n_mid_offset := (Kernel.N - 1) / 2
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Window(i - m_mid_offset, j - n_mid_offset, Kernel.M, Kernel.N).EltMul(Kernel).EltSum()
      }
   }
   return R
}

func (X *SimpleMatrix) Pool (
   pool_m, pool_n, stride_m, stride_n int, f func(float64, float64) float64, init float64,
) *SimpleMatrix {
   newM := X.M / stride_m
   if X.M % stride_m > 0 {
      newM ++
   }
   newN := X.N / stride_n
   if X.N % stride_n > 0 {
      newN ++
   }
   R := NewSimpleMatrix(newM, newN)
   m_mid_offset := (pool_m - stride_m) / 2
   n_mid_offset := (pool_n - stride_n) / 2
   for i := newM - 1; i >= 0; i-- {
      for j := newN - 1; j >= 0; j-- {
         R.Data[i][j] = X.Window(
            i * stride_m - m_mid_offset, j * stride_n - n_mid_offset, pool_m, pool_n,
         ).Reduce(f, init)
      }
   }
   return R
}

func (X *SimpleMatrix) Map (f func (float64) float64) *SimpleMatrix {
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = f(X.Data[i][j])
      }
   }
   return R
}

func (X *SimpleMatrix) Reduce (f func(float64, float64) float64, init float64) float64 {
   r := init
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         r = f(r, X.Data[i][j])
      }
   }
   return r
}

func __sum__ (r, x float64) float64 {
   return r + x
}
func (X *SimpleMatrix) EltSum () float64 {
   return X.Reduce(__sum__, 0.0)
}

func (X *SimpleMatrix) EltMax () float64 {
   return X.Reduce(math.Max, math.Inf(-1))
}

func (X *SimpleMatrix) EltMin () float64 {
   return X.Reduce(math.Min, math.Inf(1))
}

func (X *SimpleMatrix) Row (x int) *SimpleMatrix {
   R := NewSimpleMatrix(1, X.N)
   for i := X.N - 1; i >= 0; i-- {
      R.Data[0][i] = X.Data[x][i]
   }
   return R
}

func (X *SimpleMatrix) Col (x int) *SimpleMatrix {
   R := NewSimpleMatrix(X.M, 1)
   for i := X.M - 1; i >= 0; i-- {
      R.Data[i][0] = X.Data[i][x]
   }
   return R
}

func (X *SimpleMatrix) Window (y, x, h, w int) *SimpleMatrix {
   R := NewSimpleMatrix(h, w)
   m := y + h
   if m > X.M {
      m = X.M
   }
   n := x + w
   if n > X.N {
      n = X.N
   }
   for i := m - 1; i >= y; i-- {
      if i < 0 {
         continue
      }
      for j := n - 1; j >= x; j-- {
         if j < 0 {
            continue
         }
         R.Data[i - y][j - x] = X.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) MirrorM () *SimpleMatrix {
   n := X.N - 1
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Data[i][n - j]
      }
   }
   return R
}

func (X *SimpleMatrix) MirrorN () *SimpleMatrix {
   m := X.M - 1
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Data[m - i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) Dot (Y *SimpleMatrix) *SimpleMatrix {
   if X.N != Y.M {
      return nil
   }
   R := NewSimpleMatrix(X.M, Y.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := Y.N - 1; j >= 0; j-- {
         sum := 0.0
         for k := X.N - 1; k >= 0; k-- {
            sum += X.Data[i][k] * Y.Data[k][j]
         }
         R.Data[i][j] = sum
      }
   }
   return R
}

func (X *SimpleMatrix) EltMul (Y *SimpleMatrix) *SimpleMatrix {
   if X.M != Y.M && X.N != Y.N {
      return nil
   }
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Data[i][j] * Y.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) Add (Y *SimpleMatrix, a1, a2 float64) *SimpleMatrix {
   if X.M != Y.M && X.N != Y.N {
      return nil
   }
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = a1 * X.Data[i][j] + a2 * Y.Data[i][j]
      }
   }
   return R
}

func (X *SimpleMatrix) Scale (a float64) *SimpleMatrix {
   R := NewSimpleMatrix(X.M, X.N)
   for i := X.M - 1; i >= 0; i-- {
      for j := X.N - 1; j >= 0; j-- {
         R.Data[i][j] = X.Data[i][j] * a
      }
   }
   return R
}

func (X *SimpleMatrix) SacleWindow (m, n, h, w int, a float64) *SimpleMatrix {
   endM := m + h
   if endM > X.M {
      endM = X.M
   }
   if m < 0 {
      m = 0
   }
   endN := n + w
   if endN > X.N {
      endN = X.N
   }
   if m < 0 {
      m = 0
   }
   for i := endM - 1; i >= m; i-- {
      for j := endN - 1; j >= n; j-- {
         X.Data[i][j] *= a
      }
   }
   return X
}

func (X *SimpleMatrix) MapWindow (m, n, h, w int, f func (float64) float64) *SimpleMatrix {
   endM := m + h
   if endM > X.M {
      endM = X.M
   }
   if m < 0 {
      m = 0
   }
   endN := n + w
   if endN > X.N {
      endN = X.N
   }
   if m < 0 {
      m = 0
   }
   for i := endM - 1; i >= m; i-- {
      for j := endN - 1; j >= n; j-- {
         X.Data[i][j] = f(X.Data[i][j])
      }
   }
   return X
}

func (X *SimpleMatrix) ReduceWindow (
   m, n, h, w int, f func (float64, float64) float64, init float64,
) float64 {
   endM := m + h
   if endM > X.M {
      endM = X.M
   }
   if m < 0 {
      m = 0
   }
   endN := n + w
   if endN > X.N {
      endN = X.N
   }
   if m < 0 {
      m = 0
   }
   r := init
   for i := endM - 1; i >= m; i-- {
      for j := endN - 1; j >= n; j-- {
         r = f(r, X.Data[i][j])
      }
   }
   return r
}
