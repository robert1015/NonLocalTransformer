from torch.autograd import Function, Variable

class AttentionDist(Function):
    @staticmethod
    #def forward(ctx, input1, input2, y, margin):
    def forward(ctx, input, y, margin):
        ctx.margin = margin
        _output = input.clone()
        output = _output * 0.1
        ctx.save_for_backward(input , y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, y = ctx.saved_variables
        grad_input = Variable(input.data.new(input.size()).zero_())
        grad_input.add_(0.1)
        return grad_input * grad_output,None,None

