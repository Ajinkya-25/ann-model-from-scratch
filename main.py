import numpy as np

weights_total=0
class neuralnetwork:
    def __init__(self,ip_size,hidden,op_size):
        self.ip_size=ip_size
        self.hidden=hidden
        self.op_size=op_size

        self.weights_input_hidden=np.random.randn(ip_size,hidden)
        self.bias_input_hidden=np.random.randn(1)
        self.weights_hidden_output=np.random.randn(hidden,op_size)
        self.bias_hidden_output = np.array([[0.,0.,0.,0.]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self,inputs):
        self.inputs=inputs
        self.output_layer=0
        for i in inputs:
            self.hidden_layer=np.dot(i,self.weights_input_hidden)+self.bias_input_hidden
            self.hidden_output=self.sigmoid(self.hidden_layer)
            self.output_layer=self.sigmoid(np.dot(self.hidden_output,self.weights_hidden_output)+self.bias_hidden_output)
        return self.output_layer

    def backward(self, output, exp_outputs, learning_rate=0.1):
        error = exp_outputs.T - output
        delta=self.sigmoid_derivative(output)*error
        self.weights_hidden_output+=np.array([learning_rate*delta[0]*self.hidden_output]).T
        self.bias_hidden_output+=delta*learning_rate
        delta2=np.dot(delta,self.inputs)
        self.weights_input_hidden+=learning_rate*np.matmul(delta2.T,np.array([self.sigmoid_derivative(self.hidden_output)*np.sum(self.weights_hidden_output,axis=1)]))
        

    def train(self,inputs,exp_outputs, learning_rate, epochs):
        for epoch in range(epochs):
            output=self.forward(inputs)
            self.backward(output,exp_outputs, learning_rate)
            loss = np.mean(np.square(exp_outputs - output))
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            print(output)
            print(exp_outputs)
            print()

model=neuralnetwork(2,4,1)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
exp_outputs = np.array([[0], [1], [1], [0]])
model.train(inputs, exp_outputs, learning_rate=0.1, epochs=100000)