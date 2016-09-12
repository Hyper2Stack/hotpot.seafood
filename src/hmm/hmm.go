package hmm

import (
   "math"
)

type HiddenMarkovModel interface {
   N     ()                    int
   M     ()                    int
   A     (i, j int)            float64 // i,j=[0, N]
   B     (i, j int)            float64 // i=[0, N], j=[0, M]
   Pi    (i int)               float64 // i=[0, N]

   GetA  ()                    *[][]float64
   GetB  ()                    *[][]float64
   GetPi ()                    *[]float64

   SetA  (i, j int, v float64) bool
   SetB  (i, j int, v float64) bool
   SetPi (i int, v float64)    bool
   FillA (v [][]float64)
   FillB (v [][]float64)
   FillPi(v []float64)

   Scale ()                    HiddenMarkovModel
}

func ViterbiCalculator (hmm HiddenMarkovModel, observation []int) (lnP float64, state []int) {
   obn   := len(observation)
   stn   := hmm.N()
   delta := make([][]float64, obn)
   psy   := make([][]int, obn)
   for i := 0; i < obn; i++ {
      delta[i] = make([]float64, stn)
      psy[i]   = make([]int, stn)
   }
   state = make([]int, obn)
   for i := 0; i < stn; i++ {
      delta[0][i] = - math.Log(hmm.Pi(i)) - math.Log(hmm.B(i, observation[0]))
      psy[0][i]   = 0
   }

   tmp_delta := 0.0
   min_delta := 0.0
   for t := 1; t < obn; t++ {
      for from := 0; from < stn; from++ {
         // step (hmm, observation[t], t, from)
         min_delta = delta[t-1][0] - math.Log(hmm.A(0, from))
         min_psy   := 0
         for to := 1; to < stn; to++ {
            tmp_delta = delta[t-1][to] - math.Log(hmm.A(to, from))
            if tmp_delta < min_delta {
               min_delta = tmp_delta
               min_psy   = to
            }
         }
         delta[t][from] = min_delta - math.Log(hmm.B(from, observation[t]))
         psy[t][from]   = min_psy
      }
   }
   // last (hmm, observation[obn-1])
   lnP = delta[stn-1][0]
   for end := 1; end < stn; end++ {
      tmp_delta = delta[stn-1][end]
      if tmp_delta < lnP {
         lnP = tmp_delta
         state[stn-1] = end
      }
   }
   for t := stn - 2; t >= 0; t-- {
      state[t] = psy[t+1][state[t+1]]
   }
   lnP = -lnP
   return
}

func ForwardCalculator(hmm HiddenMarkovModel, observation []int) (p float64, forward [][]float64) {
   var sum float64
   obn := len(observation)
   stn := hmm.N()
   forward = make([][]float64, obn)
   for i := 0; i < obn; i++ {
      forward[i] = make([]float64, stn)
   }

   for i := 0; i < stn; i++ {
      forward[0][i] = hmm.Pi(i) * hmm.B(i, observation[0])
   }

   for t := 1; t < obn; t++ {
      for to := 0; to < stn; to++ {
         sum = 0
         for from := 0; from < stn; from++ {
            sum += forward[t-1][from] * hmm.A(from, to)
         }
         forward[t][to] = sum * hmm.B(to, observation[t])
      }
   }

   p = 0
   for i := 0; i < stn; i++ {
      p += forward[obn-1][i]
   }
   return
}

func BackwardCalculator(hmm HiddenMarkovModel, observation []int) (p float64, backward [][]float64) {
   var sum float64
   obn := len(observation)
   stn := hmm.N()
   backward = make([][]float64, obn)
   for i := 0; i < obn; i++ {
      backward[i] = make([]float64, stn)
   }

   for i := 0; i < stn; i++ {
      backward[obn-1][i] = 1
   }

   for t := obn - 2; t >= 0; t-- {
      for from := 0; from < stn; from++ {
         sum = 0
         for to := 0; to < stn; to++ {
            sum += backward[t+1][to] * hmm.A(from, to) * hmm.B(to, observation[t+1])
         }
         backward[t][from] = sum
      }
   }

   p = 0
   for i := 0; i < stn; i++ {
      p += hmm.Pi(i) * hmm.B(i, observation[0]) * backward[0][i]
   }
   return
}
