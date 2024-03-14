import numpy as np

class neuralnetwork:
    def __init__(self,ip_size,hidden,op_size):
        self.ip_size=ip_size
        self.hidden=hidden
        self.op_size=op_size

        self.weights_input_hidden=np.random.randn(ip_size,hidden)
        self.bias_input_hidden=np.random.randn(1,ip_size)
        self.weights_hidden_output=np.random.randn(hidden,op_size)
        self.bias_hidden_output = np.random.randn(1, hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self,inputs):
        hidden_layer=np.dot(inputs,self.weights_input_hidden)+self.bias_input_hidden
        hidden_output=self.sigmoid(hidden_layer)
        output_layer=np.dot(hidden_output,self.weights_hidden_output)+self.bias_hidden_output
        return output_layer

    def backward(self,inputs,outputs,learning_rate):
        error=outputs-self.output_layer

    def train(self,inputs, outputs, learning_rate, epochs):
        for epoch in range(epochs):
            output=self.forward(inputs)
            self.backward(inputs, outputs, learning_rate)
            loss = np.mean(np.square(outputs - output))
            print(f'Epoch {epoch}, Loss: {loss:.4f}')


input=np.array([[0,1],[0,0],[1,0],[1,1],[1,0],[0,0],[0,1],[1,1],[1,0],[0,0]])
output=np.array([[1],[0],[1],[1],[1],[0],[1],[1],[1],[0]])

model=neuralnetwork(2,4,1)
model.train(input,output,0.1,100)

test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print('Predictions:',model.forward(test_input))


