# easyNeuron
`easyNeuron` is a lightweight neural network framework written in Python for Python as one file.
It only uses Python Standard Library modules - not even numpy - to program it. I was inspired by **Sentdex's** nnfs project, but I came
to the end of the series so far (I got to episode 7) and was too impatient to wait for the next episode, so I created this
to learn and to share. I understand that he has released more since then, but I'll continue to develop this.

This is a very community driven project, so please report any bugs that we can iron out;
we want to make this the best we can. It is also under constant heavy development, so
please provide feedback for any issues you come accross

-------

<br/>

## **Exceptions**
`easyNeuron` uses the basic built-in exceptions that are in the Standard Library.
As well as them, `easyNeuron` has its own exceptions for comprehensive error messages to make your workflow faster.

--------
<br/>

## **Classes**

<br/>

### **Layers**
>***Layer(n_inputs, n_neurons)***
- `n_inputs` = number of inputs (from data or previous layer)
- `n_neurons` = number of neurons (will also equate to number of biases)

Parent class of all layers, containing the dunder/magic methods for all layers.

<br/>

>***Layer_Dense(n_inputs, n_neurons)***
- `n_inputs` = integer of inputs (from data or previous layer)
- `n_neurons` = integer of neurons (will also equate to number of biases)

Dense layer, fully connected.

>>*forward(inputs)*
 - `inputs` = `list` or `tuple` of data/output of previous layer

Runs the *Layer_Dense* object forwards (forwardpropagating the dense layer).

<br/>

--------

<br/>

## **Classmethods**

<br/>

> ***Matrix***

Classmethod for matrix functions such as dot products and transposing.

>>*dot(list_1, list_2)*
 - `list_1` and `list_2` = 2 lists to find the dot product between.

Returns the dot product as a `list`/`matrix` between the 2 lists. I am not using `numpy` for this, so I must write my own.

>>*transpose(matrix)*
 - `matrix` = input a `matrix` (list of lists)

Returns the matrix transposed - columns become rows and rows become columns. Another numpy method I am not using, so this is one I've written myself.

<br/>

-------

<br/>

## **Functions**

<br/>



<br/>

-------

<br/>

## **Examples**

<br/>