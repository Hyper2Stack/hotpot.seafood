package hmm

func BaumWelchLeaner (need_init bool, hmm HiddenMarkovModel, observation []int, state []int, times int) {
   obn  := len(observation)
   stn  := hmm.N()
   m    := hmm.M()
   mavg := 1.0 / float64(m)
   navg := 1.0 / float64(stn)

   if need_init {
      for i := 0; i < stn; i++ {
         for j := 0; j < stn; j++ {
            hmm.SetA(i, j, navg)
         }
         for j := 0; j < m; j++ {
            hmm.SetB(i, j, mavg)
         }
         hmm.SetPi(i, navg)
      }
   }

   // learn loop
   var s1, s2, gamman, xin float64
   newhmm := MakeBasicHMM(stn, m)
   newhmm.FillA(*hmm.GetA())
   newhmm.FillB(*hmm.GetB())
   newhmm.FillPi(*hmm.GetPi())
   for times > 0 {
      s2 = 0
      _, f := ForwardCalculator(hmm, observation)
      _, b := BackwardCalculator(hmm, observation)

      gamman = gammaN(0, stn, f, b)
      for i := 0; i < stn; i++ {
         newhmm.SetPi(i, divide0(gammaM(i, 0, f, b), gamman))
      }

      for t := 0; t < obn - 1; t++ {
         gamman = gammaN(t, stn, f, b)
         xin = xiN(hmm, t, stn, observation, f, b)
         for i := 0; i < stn; i++ {
            for j := 0; j < stn; j++ {
               newhmm.SetA(i, j,
                  divide0(newhmm.A(i, j) + xiM(hmm, i, j, t, observation, f, b), xin))
            }
            s1 = divide0(gammaM(i, t, f, b), gamman)
            newhmm.SetB(i, observation[t], newhmm.B(i, observation[t]) + s1)
            s2 += s1
         }
      }

      gamman = gammaN(obn-1, stn, f, b)
      for i := 0; i < stn; i++ {
         for j := 0; j < stn; j++ {
            newhmm.SetA(i, j, divide0(newhmm.A(i, j), s2))
         }
         s1 = divide0(gammaM(i, obn-1, f, b), gamman)
         newhmm.SetB(i, observation[obn-1], newhmm.B(i, observation[obn-1]) + s1)
         s2 += s1
      }
      for i := 0; i < stn; i++ {
         for j := 0; j < m; j++ {
            newhmm.SetB(i, j, divide0(newhmm.B(i, j), s2))
         }
      }

      hmm.FillA(*newhmm.GetA())
      hmm.FillB(*newhmm.GetB())
      hmm.FillPi(*newhmm.GetPi())
      times --
   }
}


func divide0(m, n float64) float64 {
   if n == 0 {
      return 0
   }
   return m / n
}

// gamma = gammM / gammN
func gammaM(i, t int, forward, backward [][]float64) float64 {
   return forward[i][t] * backward[i][t]
}

func gammaN(t, stn int, forward, backward [][]float64) float64 {
   var sum float64
   for j := 0; j < stn; j++ {
      sum += forward[j][t] * backward[j][t]
   }
   return sum
}

func xiM(hmm HiddenMarkovModel, i, j, t int, observation []int, forward, backward [][]float64) float64 {
   if t == len(observation) - 1 {
      return forward[i][t] * hmm.A(i, j)
   } else {
      return forward[i][t] * hmm.A(i, j) * hmm.B(j, observation[t+1]) * backward[j][t+1]
   }
}

func xiN(hmm HiddenMarkovModel, t, stn int, observation []int, forward, backward [][]float64) float64 {
   var sum, delta float64
   for i := 0; i < stn; i++ {
      for j := 0; j < stn; j++ {
         delta = forward[i][t] * hmm.A(i, j)
         if t < stn - 1 {
            delta *= hmm.B(j, observation[t+1]) * backward[j][t+1]
         }
         sum += delta
      }
   }
   return sum
}
