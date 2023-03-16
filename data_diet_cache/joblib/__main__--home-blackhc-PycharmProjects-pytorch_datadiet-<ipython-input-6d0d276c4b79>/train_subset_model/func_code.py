# first line: 1
@memory.cache()
def train_subset_model(idx, training_indices):
  # Train a model on the subset of the training data
  # Store data['train'] in a variable to avoid recomputing it
  train_original = hlb_cifar10.data["train"]
  try:
    # Set data["train"] to the subset
    hlb_cifar10.data["train"] = {
      "images": train_original["images"][training_indices],
      "targets": train_original["targets"][training_indices],
    }

    # defined by training_indices
    # Set all seeds to idx
    torch.manual_seed(idx)
    np.random.seed(idx)

    results = hlb_cifar10.main(compute_scores=False)
    return results
  finally:
    hlb_cifar10.data["train"] = train_original
