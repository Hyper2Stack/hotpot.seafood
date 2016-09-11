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
