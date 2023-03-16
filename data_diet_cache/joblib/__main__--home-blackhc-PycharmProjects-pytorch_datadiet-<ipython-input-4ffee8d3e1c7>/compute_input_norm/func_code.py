# first line: 1
@memory.cache()
def compute_input_norm():
  # flatten data["train_scores"]["input"] to (N, 3072)
  inputs = hlb_cifar10.data["train_scores"]["images"].view(-1, 3072)

  input_norm = torch.linalg.norm(inputs, ord=2, axis=1).cpu().numpy()
  return input_norm
