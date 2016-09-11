package hmm

import "testing"

const epsilon = 1e-9

func initTestData () HiddenMarkovModel {
   /*
          [0.5 0.2 0.3]      [0.5 0.5]       [0.2]
      A = [0.3 0.5 0.2]  B = [0.4 0.6]  pi = [0.4]
          [0.2 0.3 0.5]      [0.7 0.3]       [0.4]
    */
   m := MakeBasicHMM(3, 2)

   m.SetA(0, 0, 0.5); m.SetA(0, 1, 0.2); m.SetA(0, 2, 0.3)
   m.SetA(1, 0, 0.3); m.SetA(1, 1, 0.5); m.SetA(1, 2, 0.2)
   m.SetA(2, 0, 0.2); m.SetA(2, 1, 0.3); m.SetA(2, 2, 0.5)

   m.SetB(0, 0, 0.5); m.SetB(0, 1, 0.5)
   m.SetB(1, 0, 0.4); m.SetB(1, 1, 0.6)
   m.SetB(2, 0, 0.7); m.SetB(2, 1, 0.3)

   m.SetPi(0, 0.2)
   m.SetPi(1, 0.4)
   m.SetPi(2, 0.4)
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
   if delta > epsilon {
      t.Fail()
   }
}

func TestHmmViterbi (t *testing.T) {
   m := initTestData()
   _, s := ViterbiCalculator(m, []int{0, 1, 0})
   if s[0] != 2 || s[1] != 2 || s[2] != 2 {
      t.Fail()
   }
}
