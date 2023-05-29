"""
Quick and dirty reproduction of the data diet paper for CIFAR-10.
"""

import hlb_cifar10
from joblib import Memory
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# We use joblib's cache to avoid double computation
location = './data_diet_cache'
memory = Memory(location, verbose=0)

#%%

@memory.cache()
def compute_scores(idx):
  # Set all seeds to idx
  torch.manual_seed(idx)
  np.random.seed(idx)

  results = hlb_cifar10.main()
  return results


#%%

# We compute the scores for 10 different seeds
results = [compute_scores(idx) for idx in tqdm(range(10))]

#%%

# Compute the input norm for input sample in data["train_scores"]
@memory.cache()
def compute_input_norm():
  # flatten data["train_scores"]["input"] to (N, 3072)
  inputs = hlb_cifar10.data["train_scores"]["images"].view(-1, 3072)

  input_norm = torch.linalg.norm(inputs, ord=2, axis=1).cpu().numpy()
  return input_norm


input_norm_scores = compute_input_norm()

#%%

@memory.cache()
def compute_mean_el2n_scores_at_epoch_1():
  # Compute the mean el2n score for each seed
  mean_el2n_scores = np.mean([result[1][1]["el2n"] for result in results], axis=0)
  return mean_el2n_scores


mean_el2n_scores_at_epoch_1 = compute_mean_el2n_scores_at_epoch_1()

#%%

@memory.cache()
def compute_mean_grand_scores_at_init():
  # Compute the mean grand score for each seed
  mean_grand_scores = np.mean([result[1][-1]["grand"] for result in results], axis=0)
  return mean_grand_scores


mean_grand_scores_at_init = compute_mean_grand_scores_at_init()

#%%

# Compute grand scores for 100 models at init
@memory.cache()
def compute_scores_at_init(idx):
  # Set all seeds to idx
  torch.manual_seed(idx)
  np.random.seed(idx)

  results = hlb_cifar10.main(only_scores_at_init=True)
  return results

results_at_init = [compute_scores_at_init(idx) for idx in tqdm(range(100))]

#%%
del results_at_init[8]
mean_grand_scores_at_init_100 = np.mean([result[1][-1]["grand"] for result in results_at_init], axis=0)

#%%

@memory.cache()
def compute_mean_grand_scores_at_epoch_1():
  # Compute the mean grand score for each seed
  mean_grand_scores = np.mean([result[1][1]["grand"] for result in results], axis=0)
  return mean_grand_scores

mean_grand_scores_at_epoch_1 = compute_mean_grand_scores_at_epoch_1()

#%%
from copy import copy

def make_legend_lines_opaque():
  lines, labels = plt.gca().get_legend_handles_labels()

  lines = lines
  labels = labels
  lines = [copy(l) for l in lines]
  for l in lines:
    l.set_alpha(1)

  plt.legend(lines, labels, loc=0)

#%%

def plot_scores(scoreA, scoreB, scoreALabel, scoreBLabel):
  plt.figure(figsize=(8, 8/1.618))
  # Normalize the scores (x and y) separately
  #scoreA_norm = (scoreA - np.median(scoreA)) / (scoreA.max() - scoreA.min())
  #scoreB_norm = (scoreB - np.median(scoreB)) / (scoreB.max() - scoreB.min())
  scoreA_norm = (scoreA - np.min(scoreA)) / (scoreA.max() - scoreA.min())
  scoreB_norm = (scoreB - np.min(scoreB)) / (scoreB.max() - scoreB.min())
  combined_score = scoreA_norm + scoreB_norm
  # Sort the scores by combined_score
  sorted_indices = np.argsort(combined_score)
  # Plot the scores
  idx = list(range(len(scoreA)))
  plt.scatter(idx, scoreA_norm[sorted_indices], label=scoreALabel, s=1, alpha=0.1)
  plt.scatter(idx, scoreB_norm[sorted_indices], label=scoreBLabel, s=1, alpha=0.1)
  # Set title
  plt.title(f"Rank Corr.: {spearmanr(scoreA, scoreB)[0]:.2f}")
  plt.xlabel("Sorted by Avg Norm'ed Score")
  plt.ylabel("Normalized Score (Score-Min/(Max-Min))")

  make_legend_lines_opaque()

  #plt.legend()

  def label_to_filename(label):
    return label.lower().replace(" ", "_")

  plt.savefig(f'hlb_{label_to_filename(scoreALabel)}_vs_{label_to_filename(scoreBLabel)}_sorted.png', dpi=300, bbox_inches='tight', pad_inches=0,
              transparent=True)
  plt.show()

  plt.figure(figsize=(8, 8 / 1.618))

  plt.scatter(scoreA, scoreB, s=1, alpha=0.1)
  # Set title
  plt.title(f"Rank Corr.: {spearmanr(scoreA, scoreB)[0]:.2f}")
  plt.xlabel(scoreALabel)
  plt.ylabel(scoreBLabel)

  def label_to_filename(label):
    return label.lower().replace(" ", "_")

  plt.savefig(f'hlb_{label_to_filename(scoreALabel)}_vs_{label_to_filename(scoreBLabel)}_scatter.png', dpi=300,
              bbox_inches='tight', pad_inches=0,
              transparent=True)
  plt.show()

#%%

# Plot grand score vs input norm
plot_scores(input_norm_scores, mean_grand_scores_at_init, "Input Norm", "GraNd at Init")

#%%

# Plot grand score vs input norm
plot_scores(input_norm_scores, mean_grand_scores_at_init_100, "Input Norm", "GraNd at Init (100 samples)")

#%%

plot_scores(mean_el2n_scores_at_epoch_1, mean_grand_scores_at_init, "EL2N at Epoch 1", "GraNd at Init")

#%%

plot_scores(mean_el2n_scores_at_epoch_1, mean_grand_scores_at_epoch_1, "EL2N at Epoch 1", "GraNd at Epoch 1")

#%%

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


#%%

@memory.cache()
def train_random_subset_model(idx, training_fraction):
  # Train a model on a random subset of the training data
  # Set all seeds to idx
  torch.manual_seed(idx)
  np.random.seed(idx)

  # Get the number of training samples
  num_train_samples = len(hlb_cifar10.data["train"]["images"])
  # Get a random subset of the training data
  training_indices = np.random.choice(num_train_samples, size=int(training_fraction*num_train_samples), replace=False)
  results = train_subset_model(idx, training_indices)
  return results


#%%

training_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

random_subset_models = {
  training_fraction: [train_random_subset_model(idx, training_fraction) for idx in tqdm(range(10))]
                      for training_fraction in training_fractions
}
# Add results from results at training_fraction=1
random_subset_models[1.] = results

#%%

def train_score_based_subset_model(scores, idx, training_fraction):
  descending_indices = np.argsort(scores)[::-1].copy()

  return train_subset_model(idx, descending_indices[:int(training_fraction*len(descending_indices))])


#%%

subset_models_by_score_training_fraction = {
  score_type: {
    training_fraction: [
      train_score_based_subset_model(scores, idx, training_fraction) for idx in tqdm(range(10))]
      for training_fraction in training_fractions
  } for score_type, scores in [("EL2N at Epoch 1", mean_el2n_scores_at_epoch_1),
                               ("GraNd at Epoch 1", mean_grand_scores_at_epoch_1),
                               ("GraNd at Init", mean_grand_scores_at_init),
                               ("Input Norm", input_norm_scores),
                               ]
}
subset_models_by_score_training_fraction["Random"] = random_subset_models

#%%
import pandas as pd

# Turn subset_models_by_score_training_fraction into a pandas dataframe
rows = []
for score_type, models_by_training_fraction in subset_models_by_score_training_fraction.items():
  for training_fraction, models in models_by_training_fraction.items():
    for model in models:
      if training_fraction == 1.:
        for other_score_type in subset_models_by_score_training_fraction.keys():
          if other_score_type != score_type:
            rows.append({
              "Metric": other_score_type,
              "Pruned Fraction": 0,
              "Test Accuracy": model[0],
            })

      rows.append({
        "Metric": score_type,
        "Pruned Fraction": 1.-training_fraction,
        "Test Accuracy": model[0],
        })

subset_models_df = pd.DataFrame(rows)

#%%

# Seaborn plot the results by training fraction and score type
import seaborn as sns
sns.set_style("white")
# make the background opaque
#sns.set(rc={'figure.facecolor':'white', 'figure.edgecolor':'white'})

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8/1.618))

# Plot the training fraction against the accuracy
sns.set_color_codes("pastel")
sns.lineplot(x="Pruned Fraction", y="Test Accuracy", data=subset_models_df, hue="Metric",
             hue_order=["EL2N at Epoch 1", "GraNd at Epoch 1", "Random", "Input Norm", "GraNd at Init"],)
# save to svg for better quality
plt.savefig("hlb_figure11_repro_cifar10.pdf", bbox_inches="tight", pad_inches=0, transparent=True,)
plt.show()

#%%

# Creata "rank correlation" table between input norm, grad_norm, late_grad_norm, forget_scores, and l2_error_scores
import pandas as pd
import seaborn as sns

# Create a dataframe with the correlations
corr_df = pd.DataFrame({'Input Norm': input_norm_scores,
                        'GraNd at Init': mean_grand_scores_at_init,
                        'GraNd at Epoch 1': mean_grand_scores_at_epoch_1,
                        'EL2N at Epoch 1': mean_el2n_scores_at_epoch_1,
                        })


# Create a "rank correlation" table between input norm, grad_norm, late_grad_norm, forget_scores, and l2_error_scores
rcorrs = corr_df.corr(method='spearman')

# Remove all columns with column index greater row index
rcorrs = rcorrs.where(np.triu(np.ones(rcorrs.shape)).astype(bool))

#rcorrs = rcorrs.where(~np.triu(np.ones(rcorrs.shape)).astype(bool))

# Plot the heatmap
plt.figure(figsize=(8,8/1.618))
sns.heatmap(rcorrs, annot=True, cmap='coolwarm')
plt.savefig('hlb_rankcorrelation_heatmap.pdf',
             bbox_inches='tight', pad_inches=0, transparent=True)
# plt.savefig('orgrepo_rankcorrelation_heatmap.png', dpi=300,
#             bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()