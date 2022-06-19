class Always():
    val: Callable

    @nn.compact 
    def __call__(self, *args, **kwargs):
        val = self.val
        return val
      
class Residual(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        fn = self.fn
        return fn(x, **kwargs) + x
