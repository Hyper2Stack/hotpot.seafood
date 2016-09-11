package hmm

import (
   "testing"
   "fmt"
)

const epsilon = 1e-9

func initTestData () HiddenMarkovModel {
   /*
          [0.5 0.2 0.3]      [0.5 0.5]       [0.2]
      A = [0.3 0.5 0.2]  B = [0.4 0.6]  pi = [0.4]
          [0.2 0.3 0.5]      [0.7 0.3]       [0.4]
    */
   m := MakeBasicHMM(3, 2)

   m.FillA([][]float64{
      {0.5, 0.2, 0.3},
      {0.3, 0  , 0.2},
      {0.2, 0.3, 0.5},
   })
   m.SetA(1, 1, 0.5)

   m.FillB([][]float64 {
      {0.5, 0.5},
      {0.4, 0.6},
      {0  , 0.3},
   })
   m.SetB(2, 0, 0.7)

   m.FillPi([]float64{0, 0.4, 0.4})
   m.SetPi(0, 0.2)
   fmt.Println(m)
   return m
}

func TestHmmForwardAndBackward (t *testing.T) {
   m := initTestData()
   pf, _ := ForwardCalculator(m, []int{0, 1, 0})
   pb, _ := BackwardCalculator(m, []int{0, 1, 0})
   delta := pf - pb
   if delta < 0 {
      delta = -delta
   }
   fmt.Println(pf, pb, delta)
   if delta > epsilon {
      t.Fail()
   }
}

func TestHmmViterbi (t *testing.T) {
   m := initTestData()
   _, s := ViterbiCalculator(m, []int{0, 1, 0})
   fmt.Println(s)
   if s[0] != 2 || s[1] != 2 || s[2] != 2 {
      t.Fail()
   }
}
