package hmm

import (
   "sort"
)

func ExpandBasicHMM (hmm *BasicHMM, newN, newM int) *BasicHMM {
   n := hmm.N()
   m := hmm.M()
   if n >= newN && m >= newM {
      return hmm
   }
   if n > newN {
      newN = n
   }
   if m > newM {
      newM = m
   }

   newhmm := MakeBasicHMM(newN, newM)
   for i := 0; i < n; i++ {
      newhmm.SetPi(i, hmm.Pi(i))
      for j := 0; j < n; j++ {
         newhmm.SetA(i, j, hmm.A(i, j))
      }
      for j := 0; j < m; j++ {
         newhmm.SetB(i, j, hmm.B(i, j))
      }
   }
   return newhmm
}

func BasicLearner (hmm *BasicHMM, observation, state []int, importance float64) *BasicHMM {
   // should stn === obn
   stn    := len(state)
   obn    := len(observation)
   n      := basic_learner_max(state) + 1
   m      := basic_learner_max(observation) + 1
   hmm     = ExpandBasicHMM(hmm, n, m)

   if obn == 0 {
      return hmm
   }
   if importance < 0 {
      importance = 0
   }

   hmm.SetPi(state[0], hmm.Pi(state[0]) + importance)
   hmm.SetB(0, observation[0], hmm.B(0, observation[0]) + importance)
   for i := 1; i < stn; i++ {
      hmm.SetA(state[i-1], state[i], hmm.A(state[i-1], state[i]) + importance)
      hmm.SetB(state[i], observation[i], hmm.B(state[i], observation[i]) + importance)
   }

   return hmm
}

func basic_learner_uniquelen(array []int) (count int) {
   old    := int(^uint(0) >> 1)
   target := array[:]
   sort.Ints(target)
   count = 0
   for _, v := range target {
      if v != old {
         count++
         old = v
      }
   }
   return
}

func basic_learner_max(array []int) (max int) {
   max = - int(^uint(0) >> 1) - 1
   for _, v := range array {
      if max < v {
         max = v
      }
   }
   return
}

func basic_learner_divide(m, n float64) float64 {
   if n == 0 {
      return 0
   }
   return m / n
}
