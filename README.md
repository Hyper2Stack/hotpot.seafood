# Hotpot Seafood
Hotpot::Seafood - data world opening

```golang
import nn "neuralnetwork"

func main () {
   nn.RandomSeed()
   n := nn.NewNeuralChain()
   n.AddLayer(nn.NewLayerLinear(1, 2, 16, 0.5 /*weight*/, 0 /* decay */, false /* use b */))
   n.AddLayer(nn.NewLayerActivation(1, 16, "sigmoid"))
   n.AddLayer(nn.NewLayerLinear(1, 16, 1, 0.5, 0, false))
   n.AddLayer(nn.NewLayerActivation(1, 1, "sigmoid"))

   input := nn.NewSimpleMatrix(1, 2)
   input.Data[0][0] = 0
   input.Data[0][1] = 1
   expect := nn.NewSimpleMatrix(1, 1)
   expect.Data[0][0] = 1

   n.Fit(input, expect, 0.1 /* learning rate */)
   // equals
   output := n.Predict(input)
   n.Learn(output, expect)
   n.Update(0.1)
}
```

Hidden Markov Model ref: [jahmm](https://github.com/KommuSoft/jahmm)

Baum-Welch Algorithm ref: [wikipedia](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

Recurrent Nerual Network ref: [iamtrask github blog](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm)

Neural Network ref: [abll](http://people.compute.dtu.dk/~abll/blog/simple_cnn/)
