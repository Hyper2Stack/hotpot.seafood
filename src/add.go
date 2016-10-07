package main

import (
   nn "neuralnetwork"
   "math"
   "math/rand"
   "fmt"
)

func __round__ (x float64) float64 {
   if x > 0.5 {
      return 1
   } else {
      return 0
   }
}

func decodeNum (bits []float64) int {
   r := 0.0
   for i := 0; i < 8; i++ {
      r += math.Pow(2.0, float64(i)) * __round__(float64(bits[7 - i]))
   }
   return int(r)
}

func prepareBits(x int) []int {
   v := make([]int, 8)
   t := x
   for i := 0; i < 8; i++ {
      v[7 - i] = t % 2
      t /= 2
   }
   return v
}

func prepareBinaryMap() map[int][]int {
   m := make(map[int][]int)
   for i := 0; i < 256; i++ {
      m[i] = prepareBits(i)
   }
   return m
}

func main () {
   binary := prepareBinaryMap()

   nn.RandomSeed()
   n := nn.NewNeuralRecurrentChain(1, 2)
   hidden := 16
   n.AddLayer(nn.NewLayerLinear(1, 2, hidden, 0.5, 0, false))
   n.AddRecurrentLayer(nn.NewLayerActivation(1, hidden, "sigmoid"), "basic_recurrence")
   n.AddLayer(nn.NewLayerLinear(1, hidden, 1, 0.5, 0, false))
   n.AddRecurrentLayer(nn.NewLayerActivation(1, 1, "sigmoid"), "output_record")

   error := 0
   for i := 1; i <= 20000; i++ {
      a_int := rand.Intn(128)
      b_int := rand.Intn(128)
      c_int := a_int + b_int
      a := binary[a_int]
      b := binary[b_int]
      c := binary[c_int]
      in := nn.NewSimpleMatrix(1, 2)
      expect := nn.NewSimpleMatrix(1, 1)
      out := nn.NewSimpleMatrix(1, 8)
      for L := 0; L < 8; L++ {
         in.Data[0][0] = float64(a[7 - L])
         in.Data[0][1] = float64(b[7 - L])
         out.Data[0][7 - L] = n.Predict(in).Data[0][0]
      }
      for L := 8 - 1; L >= 0; L-- {
         expect.Data[0][0] = float64(c[7 - L])
         n.Learn(out.Col(7 - L), expect)
      }
      n.Update(0.1)
      if c_int != decodeNum(out.Data[0]) {
         error ++
      }

      if i % 1000 == 0 {
         fmt.Printf("error: %.2f%%\n", float64(error)/1000.0 * 100.0)
         error = 0
         fmt.Println(a, " + ", b, " = ", c, "  [A]", out.Map(__round__).Data[0])
         fmt.Println(a_int, " + ", b_int, " = ", c_int, "  [A]", decodeNum(out.Data[0]))
      }
   }

   error = 0
   for i := 1; i <= 20000; i++ {
      n.PredictRestart()
      a_int := rand.Intn(128)
      b_int := rand.Intn(128)
      c_int := a_int + b_int
      a := binary[a_int]
      b := binary[b_int]
      in := nn.NewSimpleMatrix(1, 2)
      out := nn.NewSimpleMatrix(1, 8)
      for L := 0; L < 8; L++ {
         in.Data[0][0] = float64(a[7 - L])
         in.Data[0][1] = float64(b[7 - L])
         out.Data[0][7 - L] = n.Predict(in).Data[0][0]
      }
      if c_int != decodeNum(out.Data[0]) {
         error ++
      }
   }
   fmt.Printf("Test Error: %.2f%%\n", float64(error)/20000.0 * 100.0)
}
