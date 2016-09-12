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

func TestBasicHMMExpandBasicHMM (t *testing.T) {
   m := MakeBasicHMM(3, 3)
   m.SetA(0, 0, 0.4)
   m.SetB(0, 1, 0.3)
   m.SetPi(2, 0.5)

   m = ExpandBasicHMM(m, 5, 6)
   if m.N() != 5 || m.M() != 6 {
      t.Fail()
   }
   if m.A(0, 0) != 0.4 || m.B(0, 1) != 0.3 || m.Pi(2) != 0.5 {
      t.Fail()
   }

   m = ExpandBasicHMM(m, 4, 4)
   if m.N() != 5 || m.M() != 6 {
      t.Fail()
   }
   if m.A(0, 0) != 0.4 || m.B(0, 1) != 0.3 || m.Pi(2) != 0.5 {
      t.Fail()
   }
}

func TestBasicLearner (t *testing.T) {
   m := MakeBasicHMM(0, 0)
   m = BasicLearner(m, []int{0, 1, 1}, []int{0, 1, 2}, 0.0)
   if m.N() != 3 || m.M() != 2 {
      t.Fail()
   }

   m = BasicLearner(m, []int{0, 0, 1, 1, 0, 0, 1, 0}, []int{0, 0, 1, 2, 2, 0, 1, 0}, 1.0)
   fmt.Println(m)
   if m.A(0, 0) != 1 || m.A(0, 1) != 2 || m.A(0, 2) != 0 ||
      m.A(1, 0) != 1 || m.A(1, 1) != 0 || m.A(1, 2) != 1 ||
      m.A(2, 0) != 1 || m.A(2, 1) != 0 || m.A(2, 2) != 1 ||
      m.B(0, 0) != 4 || m.B(0, 1) != 0 ||
      m.B(1, 0) != 0 || m.B(1, 1) != 2 ||
      m.B(2, 0) != 1 || m.B(2, 1) != 1 ||
      m.Pi(0) != 1 || m.Pi(1) != 0 || m.Pi(2) != 0 {
      t.Fail()
   }
}

func TestBasicHMMScale (t *testing.T) {
   m := MakeBasicHMM(3, 2)
   m.FillA([][]float64{
      {1.0, 2.0, 0.0},
      {1.0, 0.0, 1.0},
      {0.0, 0.0, 0.0},
   })
   m.FillB([][]float64{
      {4.0, 0.0},
      {0.0, 2.0},
      {1.0, 1.0},
   })
   m.FillPi([]float64{1.0, 0.0, 0.0})
   m = (m.Scale()).(*BasicHMM)
   fmt.Println(m)
   if m.A(0, 0) != 1.0 / 3.0 || m.A(0, 1) != 2.0 / 3.0 || m.A(0, 2) != 0.0 ||
      m.A(1, 0) != 1.0 / 2.0 || m.A(1, 1) != 0.0 || m.A(1, 2) != 1.0 / 2.0 ||
      m.A(2, 0) != m.A(2, 1) || m.A(2, 1) != m.A(2, 2) || m.A(2, 2) != 0.0 ||
      m.B(0, 0) != 1.0 || m.B(0, 1) != 0.0 ||
      m.B(1, 0) != 0.0 || m.B(1, 1) != 1.0 ||
      m.B(2, 0) != 0.5 || m.B(2, 1) != 0.5 ||
      m.Pi(0) != 1.0 || m.Pi(1) != m.Pi(2) || m.Pi(2) != 0.0 {
      t.Fail()
   }
}

func TestBasicHMMParse (t *testing.T) {
   m := ParseBasicHMM(`{"N":2,"M":1,"A":[[1,2],[3,4]],"B":[[5],[6]],"Pi":[7,8]}`)
   fmt.Println(m)
   if m.N() != 2 || m.M() != 1 ||
      m.A(0, 0) != 1 || m.A(0, 1) != 2 ||
      m.A(1, 0) != 3 || m.A(1, 1) != 4 ||
      m.B(0, 0) != 5 || m.B(1, 0) != 6 ||
      m.Pi(0) != 7 || m.Pi(1) != 8 {
      t.Fail()
   }
}

func TestBasicHMMStringify (t *testing.T) {
   m := MakeBasicHMM(2, 1)
   m.SetA(0, 0, 1.0)
   m.SetA(0, 1, 2.0)
   m.SetA(1, 0, 3.0)
   m.SetA(1, 1, 4.0)
   m.SetB(0, 0, 5.0)
   m.SetB(1, 0, 6.0)
   m.SetPi(0, 7.0)
   m.SetPi(1, 8.0)
   raw := StringifyBasicHMM(m)
   fmt.Println(raw)
   if raw != `{"N":2,"M":1,"A":[[1,2],[3,4]],"B":[[5],[6]],"Pi":[7,8]}` {
      t.Fail()
   }
}
