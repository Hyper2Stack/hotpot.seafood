package neuralnetwork

import (
   "math"
   "math/rand"
   "time"
)

var (
   epsilon               float64 = 1e-9
   global_guassian_cache float64 = 0.0
   global_guassian_fast  bool    = false
)

func LikeZero (x float64) bool {
   return math.Abs(x) < epsilon
}

func RandomSeed () {
   rand.Seed(time.Now().Unix())
}

func RandomLinear (a, b float64) float64 {
   return rand.Float64() * (b - a) + a
}

func RandomStandardGuassian () float64 {
   // ref: https://github.com/karpathy/recurrentjs
   if global_guassian_fast {
      global_guassian_fast = false
      return global_guassian_cache
   }
   u := 0.0
   v := 0.0
   m := 0.0
   for m == 0.0 || m > 1.0 {
      u = 2 * rand.Float64() - 1
      v = 2 * rand.Float64() - 1
      m = u * u + v * v
   }
   m = math.Sqrt(-2 * math.Log(m) / m)
   global_guassian_cache = v * m
   global_guassian_fast = true
   return u * m
}

func RandomGuassian (mu, std float64) float64 {
   return mu + RandomStandardGuassian() * std
}

func Sigmoid (x float64) float64 {
   return 1.0/(1.0 + math.Exp(x))
}

func SigmoidDerivative (x float64) float64 {
   return x * (1.0 - x)
}

func Tanh (x float64) float64 {
   return math.Tanh(x)
}

func TanhDerivative (x float64) float64 {
   return (1.0 - x * x)
}

func Relu (x float64) float64 {
   return math.Max(0, x)
}

func ReluDerivative(x float64) float64 {
   if x > 0 {
      return x
   }
   return 0.0
}
