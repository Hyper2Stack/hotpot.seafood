package hmm

type BasicHMM struct {
   HiddenMarkovModel
   n  int
   m  int
   a  [][]float64
   b  [][]float64
   pi []float64
}

func MakeBasicHMM (n, m int) *BasicHMM {
   p := new(BasicHMM)
   p.n = n
   p.m = m
   p.a = make([][]float64, n)
   p.b = make([][]float64, n)
   p.pi =  make([]float64, n)
   for i := 0; i < n; i ++ {
      p.a[i] = make([]float64, n)
   }
   for i := 0; i < n; i ++ {
      p.b[i] = make([]float64, m)
   }
   return p
}

func (h *BasicHMM) N () int {
   return h.n
}

func (h *BasicHMM) M () int {
   return h.m
}

func (h *BasicHMM) A (i, j int) float64 {
   return h.a[i][j]
}

func (h *BasicHMM) B (i, j int) float64 {
   return h.b[i][j]
}

func (h *BasicHMM) Pi (i int) float64 {
   return h.pi[i]
}

func (h *BasicHMM) GetA () *[][]float64 {
   return &h.a
}

func (h *BasicHMM) GetB () *[][]float64 {
   return &h.b
}

func (h *BasicHMM) GetPi () *[]float64 {
   return &h.pi
}

func (h *BasicHMM) FillA (v [][]float64) {
   n := h.N()
   for i := 0; i < n; i++ {
      for j := 0; j < n; j++ {
         h.SetA(i, j, v[i][j])
      }
   }
}

func (h *BasicHMM) FillB (v [][]float64) {
   n := h.N()
   m := h.M()
   for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
         h.SetB(i, j, v[i][j])
      }
   }
}

func (h *BasicHMM) FillPi (v []float64) {
   n := h.N()
   for i := 0; i < n; i++ {
      h.SetPi(i, v[i])
   }
}

func (h *BasicHMM) SetA (i, j int, v float64) bool {
   h.a[i][j] = v
   return true
}

func (h *BasicHMM) SetB (i, j int, v float64) bool {
   h.b[i][j] = v
   return true
}

func (h *BasicHMM) SetPi (i int, v float64) bool {
   h.pi[i] = v
   return true
}

func (h *BasicHMM) Scale () HiddenMarkovModel {
   n      := h.N()
   m      := h.M()
   newhmm := MakeBasicHMM(n, m)
   newhmm.FillA(*h.GetA())
   newhmm.FillB(*h.GetB())
   newhmm.FillPi(*h.GetPi())

   var s1, s2, s3 float64
   s3 = 0
   for i := 0; i < n; i++ {
      s1 = 0
      s2 = 0
      s3 += newhmm.Pi(i)
      for j := 0; j < n; j++ {
         s1 += newhmm.A(i, j)
      }
      for j := 0; j < m; j++ {
         s2 += newhmm.B(i, j)
      }

      if s1 == 0.0 {
         s1 = 1.0
      }
      if s2 == 0.0 {
         s2 = 1.0
      }
      for j := 0; j < n; j++ {
         newhmm.SetA(i, j, newhmm.A(i, j) / s1)
      }
      for j := 0; j < m; j++ {
         newhmm.SetB(i, j, newhmm.B(i, j) / s2)
      }
   }
   for i := 0; i < n; i++ {
      newhmm.SetPi(i, newhmm.Pi(i) / s3)
   }
   return newhmm
}
