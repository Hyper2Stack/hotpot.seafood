package main

import (
   "encoding/json"
   "io/ioutil"

   nn "neuralnetwork"
   "math/rand"
   "fmt"
)

type MNISTDataset struct {
   LearnSet [][]float64 `json:"learnset"`
   LearnLab []float64   `json:"learnlabel"`
   TestSet  [][]float64 `json:"testset"`
   TestLab  []float64   `json:"testlabel"`
}

// ref: http://yann.lecun.com/exdb/mnist/
// 0.01 means sample 1% of MNIST dataset (train:600, test:100)
func MNISTDataLoad () *MNISTDataset {
   fmt.Println("Loading MNIST dataset ...")
   raw, err := ioutil.ReadFile("src/mnist-0.01.json")
   if err != nil {
      fmt.Println("Load json file \"mnist-0.01.json\" failed.")
      return nil
   }
   r := new(MNISTDataset)
   json.Unmarshal(raw, r)
   for _, image := range r.LearnSet {
      for i := len(image) - 1; i >= 0; i-- {
         image[i] *= 1.0 / 255.0
      }
   }
   for _, image := range r.TestSet {
      for i := len(image) - 1; i >= 0; i-- {
         image[i] *= 1.0 / 255.0
      }
   }
   fmt.Println("Loaded.")
   return r
}

func LabelEncodeVector (klass float64) *nn.SimpleMatrix {
   klass_vec := make([]float64, 10)
   klass_vec[int(klass)] = 1.0
   return nn.NewSimpleMatrix(1, 10).FillElt(klass_vec)
}

func LabelDecode (X *nn.SimpleMatrix) float64 {
   max := X.EltMax()
   for i, v := range X.Data[0] {
      if v == max {
         return float64(i)
      }
   }
   return -1
}

func LabelEqual (predict, expect *nn.SimpleMatrix) bool {
   return LabelDecode(predict) == LabelDecode(expect)
}

func __round__ (x float64) float64 {
   if x > 0.5 {
      return 1.0
   }
   return 0.0
}

func mnist (dataset *MNISTDataset) {
   nn.RandomSeed()
   n := nn.NewNeuralChain()

   n.AddLayer(nn.NewLayerConvolution(
      /* 1*1 image */ 1, 1, /* 12 filters */ 12,
      /* 28*28 pixels */ 28, 28,
      /* 5*5 kernel */ 5, 5,
      /* decay */ 0.001))
   n.AddLayer(nn.NewLayerActivation(1* 28, 12 * 28, "tanh"))
   n.AddLayer(nn.NewLayerPoolMax(1, 12, 28, 28, 2, 2))
   n.AddLayer(nn.NewLayerConvolution(1, 12, 16, 14, 14, 5, 5, 0.001))
   n.AddLayer(nn.NewLayerActivation(1 * 14, 16 * 14, "tanh"))
   n.AddLayer(nn.NewLayerFlatten(1 * 14, 16 * 14))
   n.AddLayer(nn.NewLayerLinear(1, 16 * 14 * 14, 10, 0.5, 0, true))
   n.AddLayer(nn.NewLayerLogRegression(1, 10))

   error := 0
   for i := 1; i <= 10000; i++ {
      k := rand.Intn(600)
      image := nn.NewSimpleMatrix(28, 28).FillElt(dataset.LearnSet[k])
      label := LabelEncodeVector(dataset.LearnLab[k])
      predict := n.Predict(image)
      n.Learn(predict, label)
      n.Update(0.1)

      if !LabelEqual(predict, label) {
         error ++
      }
      if i % 1000 == 0 {
         fmt.Printf("error: %.2f%%\n", float64(error) / 1000.0 * 100.0)
         error = 0
         fmt.Println("Image", k, "->", dataset.LearnLab[k], label.Data[0], "  [A]", LabelDecode(predict), predict.Data[0])
      }
   }

   error = 0
   for i := 1; i <= 100; i++ {
      k := i - 1
      image := nn.NewSimpleMatrix(28, 28).FillElt(dataset.TestSet[k])
      label := LabelEncodeVector(dataset.TestLab[k])
      predict := n.Predict(image)
      if !LabelEqual(predict, label) {
         error ++
      }
   }
   fmt.Printf("Test Error: %.2f%%\n", float64(error))
}

func main () {
   mnist(MNISTDataLoad())
}
