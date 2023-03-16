# first line: 1
@memory.cache()
def compute_scores_at_init(idx):
  # Set all seeds to idx
  torch.manual_seed(idx)
  np.random.seed(idx)

  results = hlb_cifar10.main(only_scores_at_init=True)
  return results
