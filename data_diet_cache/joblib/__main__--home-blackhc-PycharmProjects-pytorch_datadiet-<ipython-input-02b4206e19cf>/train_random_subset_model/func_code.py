# first line: 1
@memory.cache()
def train_random_subset_model(idx, training_fraction):
  # Train a model on a random subset of the training data
  # Set all seeds to idx
  torch.manual_seed(idx)
  np.random.seed(idx)

  # Get the number of training samples
  num_train_samples = len(hlb_cifar10.data["train_original"]["input"])
  # Get a random subset of the training data
  training_indices = np.random.choice(num_train_samples, size=int(training_fraction*num_train_samples), replace=False)
  results = train_subset_model(idx, training_indices)
  return results
