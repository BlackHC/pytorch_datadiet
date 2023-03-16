# first line: 1
@memory.cache()
def compute_mean_el2n_scores_at_epoch_1():
  # Compute the mean el2n score for each seed
  mean_el2n_scores = np.mean([result[1][1]["el2n"] for result in results], axis=0)
  return mean_el2n_scores
