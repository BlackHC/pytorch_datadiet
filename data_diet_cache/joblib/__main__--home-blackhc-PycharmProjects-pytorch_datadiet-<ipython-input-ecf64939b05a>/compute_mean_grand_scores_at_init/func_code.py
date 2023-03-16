# first line: 1
@memory.cache()
def compute_mean_grand_scores_at_init():
  # Compute the mean grand score for each seed
  mean_grand_scores = np.mean([result[1][-1]["grand"] for result in results], axis=0)
  return mean_grand_scores
